#include <image_representation/ImageRepresentation.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <std_msgs/Float32.h>

#include <glog/logging.h>
#include <omp.h>


#include <cmath>
#include <vector>

// #define ESVIO_REPRESENTATION_LOG

namespace image_representation
{
  ImageRepresentation::ImageRepresentation(ros::NodeHandle &nh, ros::NodeHandle nh_private) : nh_(nh)
  {
    pub_flag_ = false;
    // setup subscribers and publishers
    sync_topic_ = nh_.subscribe("sync", 1, &ImageRepresentation::syncCallback, this);
    event_sub_ = nh_.subscribe("events", 0, &ImageRepresentation::eventsCallback, this);
    image_transport::ImageTransport it_(nh_);
    nh_private.param<bool>("is_left", is_left_, true);    // is left camera
    
    
    // voxel_pub_ = nh.advertise<dvs_msgs::TimedFloat32MultiArray>("voxel_grid", 10);
    voxel_pub_ = nh.advertise<sdevo_msgs::TimedFloat32MultiArray>("voxel_grid", 10);
    nh_private.param<bool>("use_sim_time", bUse_Sim_Time_, true);

    // system variables
    int representation_mode;
    nh_private.param<int>("representation_mode", representation_mode, 0);
    nh_private.param<int>("max_event_num_per_frame", max_event_num_per_frame_, 2000000);
    nh_private.param<int>("max_event_queue_len", max_event_queue_length_, 20);
    nh.param<bool>("/isTUMDataset", isTUMDataset_, false);

    std::cout << "isTUMDataset: " << isTUMDataset_ << std::endl;
    
    representation_mode_ = (RepresentationMode)representation_mode;
       
    // rectify variables
    bCamInfoAvailable_ = false;
    bSensorInitialized_ = false;
    sensor_size_ = cv::Size(0, 0);

    // local parameters
    nh_private.param<double>("duration_ms", duration_ms_, 100.0);
    nh_private.param<bool>("use_stereo_cam", bUseStereoCam_, true);
    nh_private.param<double>("generation_rate_hz", generation_rate_hz_, 100);
    nh_private.param("calibInfoDir", calibInfoDir_, std::string("path is not given"));
    if (!loadCalibInfo(calibInfoDir_, is_left_))
    {
      ROS_ERROR("Load Calib Info Error!!!  Given path is: %s", calibInfoDir_.c_str());
    }
    first_event_time_ = -1.0;

    if(is_left_)
      LOG(INFO) << "\33[32m" << "Left event representation node is up " << "\33[0m";
    else
      LOG(INFO) << "\33[32m" << "Right event representation node is up " << "\33[0m";

  }

  ImageRepresentation::~ImageRepresentation()
  {
    voxel_pub_.shutdown();
  }

  void ImageRepresentation::init(int width, int height)
  {
    sensor_size_ = cv::Size(width, height);
    bSensorInitialized_ = true;
    ROS_INFO("Sensor size: (%d x %d)", sensor_size_.width, sensor_size_.height);
    vEvents_.reserve(100000000);
  }

  void ImageRepresentation::syncCallback(const std_msgs::TimeConstPtr& msg)
  {
    TicToc t;
    if(pub_flag_)
    {
      voxel_pub_.publish(timed_voxel_msg_);
      pub_flag_ = false;
    }
      
    if(bUse_Sim_Time_)
      sync_time_ = ros::Time::now();
    else
      sync_time_ = msg->data;
    if (sync_time_ > ros::Time(0.1)) 
    {
      createImageRepresentationAtTime(sync_time_);
    }

  }

  void ImageRepresentation::remove_hot_pixels(std::vector<float>& voxel, int num_hot_pixels, int num_stds) 
  {
    std::cout <<"num_hot_pixels: " << num_hot_pixels << std::endl;
    std::cout <<"num_stds: " << num_stds << std::endl;
    assert(voxel.size() > 0);  

    float mean = 0.0f;
    float m2 = 0.0f;
    size_t n = voxel.size();
    
    for (size_t i = 0; i < n; ++i) {
        float value = voxel[i];
        float delta = value - mean;
        mean += delta / (i + 1);  
        m2 += delta * (value - mean);  
    }

    float std_dev = std::sqrt(m2 / n);

    if (num_hot_pixels != -1) {
      std::vector<int> indices(voxel.size());
      for (int i = 0; i < voxel.size(); ++i) {
          indices[i] = i;
      }

      std::sort(indices.begin(), indices.end(), [&voxel](int i, int j) {
          return voxel[i] > voxel[j];  
      });

      for (int i = 0; i < num_hot_pixels; ++i) {
          voxel[indices[i]] = 0.0f; 
      }
    } else {
      float threshold = mean + num_stds * std_dev;

      for (int i = 0; i < voxel.size(); ++i) {
          if (std::abs(voxel[i]) > threshold) {
              voxel[i] = 0.0f; 
          }
      }
    }
  }


  bool ImageRepresentation::toVoxelGrid(
      std_msgs::Float32MultiArray &voxel_msg,
      std::vector<dvs_msgs::Event>::iterator &ptr_b, int distance, int nb_of_time_bins)
  {
      int num_used_events = 0;
      int H = sensor_size_.height;
      int W = sensor_size_.width;
      voxel_msg.data.resize(nb_of_time_bins * H * W, 0.0f);

      double duration = duration_ms_ * 1000.0;
      double start_timestamp = ptr_b->ts.toNSec() / 1000.0; 

      int max_event_num = max_event_num_per_frame_;
      double sampling_rate = distance > max_event_num ? (double)distance / (double)max_event_num : 1.0;
      double curr_num = 0.0;
      double last_num = 0.0;

      for (int i = 0; i < distance; i += 1)
      {
          curr_num = 1 + curr_num;
          if(curr_num - last_num < sampling_rate)
          {
            ptr_b++;
            continue;
          }
          last_num = last_num + sampling_rate;
          const dvs_msgs::Event &e = *ptr_b;
          
          Eigen::Matrix<double, 2, 1> undistorted_coords = getRectifiedUndistortedCoordinate(e.x, e.y);
          float x = static_cast<float>(undistorted_coords(0));
          float y = static_cast<float>(undistorted_coords(1));
          double t = ((e.ts.toNSec() / 1000.0) - start_timestamp) * (nb_of_time_bins - 1) / duration;
          int polarity = e.polarity == 0 ? -1 : 1;
          
          int left_x = std::floor(x);
          int right_x = left_x + 1;
          int left_y = std::floor(y);
          int right_y = left_y + 1;
          int left_t = std::floor(t);
          int right_t = left_t + 1;
          

          for (int lim_x : {left_x, right_x})
          {
              if (lim_x < 0 || lim_x >= W) continue; 

              for (int lim_y : {left_y, right_y})
              {
                  if (lim_y < 0 || lim_y >= H) continue; 
                  for (int lim_t : {left_t, right_t})
                  {
                      if (lim_t < 0 || lim_t >= nb_of_time_bins) continue;

                      float weight = polarity *
                                    (1.0d - std::abs(lim_x - x)) *
                                    (1.0d - std::abs(lim_y - y)) *
                                    (1.0d - std::abs(lim_t - t));

                      voxel_msg.data[lim_t * H * W + lim_y * W + lim_x] += weight; 
                  }
            }
          }
          num_used_events++;
          ptr_b++;  
      }
      if(isTUMDataset_)
        remove_hot_pixels(voxel_msg.data, -1, 10);


      return true;


  }



  void ImageRepresentation::createImageRepresentationAtTime(const ros::Time &external_sync_time)
  { 
    if (!bcreat_)
      return;
    else
      bcreat_ = false;
      
    if (vEvents_.size() == 0)
      return;
    
    if (representation_mode_ == Voxel)
    {
      if (first_event_time_ < 0) 
      {
        std::cout << "===========================" << std::endl;
        std::cout << std::fixed << "Before update - first_event_time_: " << first_event_time_ << "  = " << vEvents_.begin()->ts.toSec() << std::endl;
        first_event_time_ = vEvents_.begin()->ts.toSec();
        std::cout << std::fixed << "After update - first_event_time_: " << first_event_time_ << "  = " << vEvents_.begin()->ts.toSec() << std::endl;
      }
        
      if(vEvents_.back().ts.toSec() - first_event_time_ < (duration_ms_ / 1000.0 * 1.1))
        return;
      int distance = 0;
      double external_t_end = first_event_time_ + duration_ms_ / 1000.0;
      double external_t_begin = first_event_time_;

      std::vector<dvs_msgs::Event>::iterator ptr_e, ptr_b, ptr_next;
      std::vector<dvs_msgs::Event>::iterator ptr_ts_b, ptr_ts_e;
      {
        std::lock_guard<std::mutex> lock(data_mutex_);
        if (!bSensorInitialized_ || !bCamInfoAvailable_)
          return;
        
        first_event_time_ = first_event_time_ + (1.0 / generation_rate_hz_);
        ptr_next = EventVector_lower_bound(vEvents_, first_event_time_);

        ptr_e = EventVector_lower_bound(vEvents_, external_t_end);
        ptr_b = EventVector_lower_bound(vEvents_, external_t_begin);
        
        distance = std::distance(ptr_b, ptr_e);
        selectedEvents_.assign(ptr_b, ptr_e);
        clearEvents(distance, ptr_next);
        
      }
    
      ptr_b = selectedEvents_.begin();
      distance = selectedEvents_.size();

      int depth = 5;
      std_msgs::Float32MultiArray voxel_msg;
      
      voxel_msg.layout.dim.push_back(std_msgs::MultiArrayDimension());
      voxel_msg.layout.dim.push_back(std_msgs::MultiArrayDimension());
      voxel_msg.layout.dim.push_back(std_msgs::MultiArrayDimension());

      voxel_msg.layout.dim[0].label = "depth";
      voxel_msg.layout.dim[0].size = depth; // nb_of_time_bins
      voxel_msg.layout.dim[0].stride = depth * sensor_size_.height * sensor_size_.width;

      voxel_msg.layout.dim[1].label = "height";
      voxel_msg.layout.dim[1].size = sensor_size_.height; // H
      voxel_msg.layout.dim[1].stride = sensor_size_.height * sensor_size_.width;

      voxel_msg.layout.dim[2].label = "width";
      voxel_msg.layout.dim[2].size = sensor_size_.width; // W
      voxel_msg.layout.dim[2].stride = sensor_size_.width;

      timed_voxel_msg_.header.stamp.fromSec(ptr_b->ts.toSec() + (0.5 * duration_ms_ / 1000.0)); 
      toVoxelGrid(voxel_msg, ptr_b, distance, depth);

      timed_voxel_msg_.data = voxel_msg; 
      pub_flag_ = true;
    }
    
  }

  void ImageRepresentation::clearEvents(int distance, std::vector<dvs_msgs::Event>::iterator ptr_e)
  {
    if (vEvents_.size() > distance + 2)
      vEvents_.erase(vEvents_.begin(), ptr_e);
    else
      vEvents_.clear();
  }

  void ImageRepresentation::eventsCallback(const dvs_msgs::EventArray::ConstPtr &msg)
  {
    TicToc t;
    std::lock_guard<std::mutex> lock(data_mutex_);
    double t1 = t.toc();
    if (!bSensorInitialized_)
      init(msg->width, msg->height);
    for (const dvs_msgs::Event &e : msg->events)
    {
      if (e.x > sensor_size_.width || e.y > sensor_size_.height)
        continue;
      vEvents_.push_back(e);

      int i = vEvents_.size() - 2;
      while (i >= 0 && vEvents_[i].ts > e.ts)
      {
        vEvents_[i + 1] = vEvents_[i];
        i--;
      }
      vEvents_[i + 1] = e;
    }
    clearEventQueue();
    bcreat_ = true;
  }

  void ImageRepresentation::clearEventQueue()
  {
    static constexpr size_t MAX_EVENT_QUEUE_LENGTH = 50000000;
    if (vEvents_.size() > MAX_EVENT_QUEUE_LENGTH)
    {
      size_t remove_events = vEvents_.size() - MAX_EVENT_QUEUE_LENGTH;
      vEvents_.erase(vEvents_.begin(), vEvents_.begin() + remove_events);
    }
  }

  bool ImageRepresentation::loadCalibInfo(const std::string &cameraSystemDir, bool &is_left)
  {
    bCamInfoAvailable_ = false;
    std::string cam_calib_dir;
    if (is_left)
      cam_calib_dir = cameraSystemDir + "/left.yaml";
    else
      cam_calib_dir = cameraSystemDir + "/right.yaml";
    if (!fileExists(cam_calib_dir))
      return bCamInfoAvailable_;
    YAML::Node CamCalibInfo = YAML::LoadFile(cam_calib_dir);

    // load calib (left)
    size_t width = CamCalibInfo["image_width"].as<int>();
    size_t height = CamCalibInfo["image_height"].as<int>();
    std::string cameraNameLeft = CamCalibInfo["camera_name"].as<std::string>();
    std::string distortion_model = CamCalibInfo["distortion_model"].as<std::string>();
    std::vector<double> vD, vK, vRectMat, vP;
    std::vector<double> vT_right_left, vT_b_c;

    vD = CamCalibInfo["distortion_coefficients"]["data"].as<std::vector<double>>();
    vK = CamCalibInfo["camera_matrix"]["data"].as<std::vector<double>>();
    vRectMat = CamCalibInfo["rectification_matrix"]["data"].as<std::vector<double>>();
    vP = CamCalibInfo["projection_matrix"]["data"].as<std::vector<double>>();

    // vT_right_left = CamCalibInfo["T_right_left"]["data"].as<std::vector<double>>();
    // vT_b_c = CamCalibInfo["T_b_c"]["data"].as<std::vector<double>>();

    cv::Size sensor_size(width, height);
    camera_matrix_ = cv::Mat(3, 3, CV_64F);
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        camera_matrix_.at<double>(cv::Point(i, j)) = vK[i + j * 3];

    distortion_model_ = distortion_model;
    dist_coeffs_ = cv::Mat(vD.size(), 1, CV_64F);
    for (int i = 0; i < vD.size(); i++)
      dist_coeffs_.at<double>(i) = vD[i];

    if (bUseStereoCam_)
    {
      rectification_matrix_ = cv::Mat(3, 3, CV_64F);
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
          rectification_matrix_.at<double>(cv::Point(i, j)) = vRectMat[i + j * 3];

      projection_matrix_ = cv::Mat(3, 4, CV_64F);
      for (int i = 0; i < 4; i++)
        for (int j = 0; j < 3; j++)
          projection_matrix_.at<double>(cv::Point(i, j)) = vP[i + j * 4];

      if (distortion_model_ == "equidistant")
      {
        cv::fisheye::initUndistortRectifyMap(camera_matrix_, dist_coeffs_,
                                             rectification_matrix_, projection_matrix_,
                                             sensor_size, CV_32FC1, undistort_map1_, undistort_map2_);
        bCamInfoAvailable_ = true;
        ROS_INFO("Camera information is loaded (Distortion model %s).", distortion_model_.c_str());
      }
      else if (distortion_model_ == "plumb_bob")
      {
        cv::initUndistortRectifyMap(camera_matrix_, dist_coeffs_,
                                    rectification_matrix_, projection_matrix_,
                                    sensor_size, CV_32FC1, undistort_map1_, undistort_map2_);
        bCamInfoAvailable_ = true;
        ROS_INFO("Camera information is loaded (Distortion model %s).", distortion_model_.c_str());
      }
      else
      {
        ROS_ERROR_ONCE("Distortion model %s is not supported.", distortion_model_.c_str());

        return bCamInfoAvailable_;
      }

      /* pre-compute the undistorted-rectified look-up table */
      precomputed_rectified_points_ = Eigen::Matrix2Xd(2, sensor_size.height * sensor_size.width);
      // raw coordinates
      cv::Mat_<cv::Point2f> RawCoordinates(1, sensor_size.height * sensor_size.width);
      for (int y = 0; y < sensor_size.height; y++)
      {
        for (int x = 0; x < sensor_size.width; x++)
        {
          int index = y * sensor_size.width + x;
          RawCoordinates(index) = cv::Point2f((float)x, (float)y);
        }
      }
      // undistorted-rectified coordinates
      cv::Mat_<cv::Point2f> RectCoordinates(1, sensor_size.height * sensor_size.width);
      if (distortion_model_ == "plumb_bob")
      {
        cv::undistortPoints(RawCoordinates, RectCoordinates, camera_matrix_, dist_coeffs_,
                            rectification_matrix_, projection_matrix_);
        ROS_INFO("Undistorted-Rectified Look-Up Table with Distortion model: %s", distortion_model_.c_str());
      }
      else if (distortion_model_ == "equidistant")
      {
        cv::fisheye::undistortPoints(
            RawCoordinates, RectCoordinates, camera_matrix_, dist_coeffs_,
            rectification_matrix_, projection_matrix_);
        ROS_INFO("Undistorted-Rectified Look-Up Table with Distortion model: %s", distortion_model_.c_str());
      }
      else
      {
        ROS_INFO("Unknown distortion model is provided.");
        return bCamInfoAvailable_;
      }
      // load look-up table
      for (size_t i = 0; i < sensor_size.height * sensor_size.width; i++)
      {
        precomputed_rectified_points_.col(i) = Eigen::Matrix<double, 2, 1>(
            RectCoordinates(i).x, RectCoordinates(i).y);
      }
      ROS_INFO("Undistorted-Rectified Look-Up Table has been computed.");
    }
    else
    {
      // TODO: calculate undistortion map
      bCamInfoAvailable_ = true;
    }
    return bCamInfoAvailable_;
  }

  Eigen::Matrix<double, 2, 1>
  ImageRepresentation::getRectifiedUndistortedCoordinate(int xcoor, int ycoor)
  {
    size_t index = ycoor * sensor_size_.width + xcoor;
    return precomputed_rectified_points_.block<2, 1>(0, index);
  }

  bool ImageRepresentation::fileExists(const std::string &filename)
  {
    std::ifstream file(filename);
    return file.good();
  }

} // namespace image_representation
