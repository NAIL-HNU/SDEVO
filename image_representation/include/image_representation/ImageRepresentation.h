#ifndef image_representation_H_
#define image_representation_H_

#include <ros/ros.h>
#include <std_msgs/Time.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>
#include <dynamic_reconfigure/server.h>
#include <image_transport/image_transport.h>
#include <image_representation/TicToc.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>
#include <std_msgs/Float32MultiArray.h>
#include <sdevo_msgs/TimedFloat32MultiArray.h>

#include <deque>
#include <mutex>
#include <Eigen/Eigen>
#include <vector>
#include <algorithm>
#include <thread>

#include <yaml-cpp/yaml.h>

namespace image_representation
{
  using EventQueue = std::deque<dvs_msgs::Event>;

  struct ROSTimeCmp
  {
    bool operator()(const ros::Time &a, const ros::Time &b) const
    {
      return a.toNSec() < b.toNSec();
    }
  };
  using GlobalEventQueue = std::map<ros::Time, dvs_msgs::Event, ROSTimeCmp>;

  inline static EventQueue::iterator EventBuffer_lower_bound(
      EventQueue &eb, ros::Time &t)
  {
    return std::lower_bound(eb.begin(), eb.end(), t,
                            [](const dvs_msgs::Event &e, const ros::Time &t)
                            { return e.ts.toSec() < t.toSec(); });
  }

  inline static EventQueue::iterator EventBuffer_upper_bound(
      EventQueue &eb, ros::Time &t)
  {
    return std::upper_bound(eb.begin(), eb.end(), t,
                            [](const ros::Time &t, const dvs_msgs::Event &e)
                            { return t.toSec() < e.ts.toSec(); });
  }

  inline static std::vector<dvs_msgs::Event>::iterator EventVector_lower_bound(
      std::vector<dvs_msgs::Event> &ev, double &t)
  {
    return std::lower_bound(ev.begin(), ev.end(), t,
                            [](const dvs_msgs::Event &e, const double &t)
                            { return e.ts.toSec() < t; });
  }

  inline static std::vector<dvs_msgs::Event>::iterator EventVector_upper_bound(
      std::vector<dvs_msgs::Event> &ev, double &t)
  {
    return std::upper_bound(ev.begin(), ev.end(), t,
                            []( const double &t, const dvs_msgs::Event &e)
                            { return t < e.ts.toSec(); });
  }

  class ImageRepresentation
  {
  public:
    ImageRepresentation(ros::NodeHandle &nh, ros::NodeHandle nh_private);
    virtual ~ImageRepresentation();

    static bool compare_time(const dvs_msgs::Event &e, const double reference_time)
    {
      return reference_time < e.ts.toSec();
    }

  private:
    ros::NodeHandle nh_;
    // core
    void init(int width, int height);
    void createImageRepresentationAtTime(const ros::Time &external_sync_time);

    // callbacks
    void eventsCallback(const dvs_msgs::EventArray::ConstPtr &msg);
    void syncCallback(const std_msgs::TimeConstPtr& msg);

    // utils
    void clearEventQueue();
    bool loadCalibInfo(const std::string &cameraSystemDir, bool &is_left);
    Eigen::Matrix<double, 2, 1>  getRectifiedUndistortedCoordinate(int xcoor, int ycoor);
    void clearEvents(int distance, std::vector<dvs_msgs::Event>::iterator ptr_e);

    void remove_hot_pixels(std::vector<float>& voxel, int num_hot_pixels = -1, int num_stds = 10); 

    bool toVoxelGrid(std_msgs::Float32MultiArray &voxel_msg, std::vector<dvs_msgs::Event>::iterator &ptr_e, int distance, int nb_of_time_bins);

    bool fileExists(const std::string &filename);
    // tests

    // calibration parameters
    cv::Mat camera_matrix_, dist_coeffs_;
    cv::Mat rectification_matrix_, projection_matrix_;
    std::string distortion_model_;
    cv::Mat undistort_map1_, undistort_map2_;
    Eigen::Matrix2Xd precomputed_rectified_points_;

    // sub & pub
    ros::Subscriber event_sub_;
    ros::Subscriber sync_topic_;

    ros::Publisher voxel_pub_;
    sdevo_msgs::TimedFloat32MultiArray timed_voxel_msg_;
    bool pub_flag_;
    bool left_;

    // online parameters
    bool bCamInfoAvailable_;
    bool bUse_Sim_Time_;
    cv::Size sensor_size_;
    ros::Time sync_time_;
    bool bSensorInitialized_;
    double first_event_time_;

    // offline parameters TODO

    int max_event_queue_length_;
    int events_maintained_size_;
    int max_event_num_per_frame_;

    // containers
    EventQueue events_;

    std::vector<dvs_msgs::Event> vEvents_, selectedEvents_;

    bool is_left_, bcreat_, isTUMDataset_;

    // thread mutex
    std::mutex data_mutex_;

    enum RepresentationMode
    {
      Voxel,     //0
    } representation_mode_;

    // parameters
    double duration_ms_;
    bool bUseStereoCam_;
    double generation_rate_hz_;
    // std::vector<dvs_msgs::Event>::iterator ptr_e_;

    // calib info
    std::string calibInfoDir_;
    std::vector<cv::Point> trapezoid_;
  };
} // namespace image_representation
#endif // image_representation_H_
