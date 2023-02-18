// MIT License

#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <nav_msgs/Odometry.h>
#include <gazebo_msgs/LinkStates.h>
#include <ros/ros.h>


void link_states_callback(gazebo_msgs::LinkStates ls_msg);

geometry_msgs::TransformStamped odom_trans, body_to_pose;
nav_msgs::Odometry odom_msg;
std::string map_frame, body_frame, gaz_body_link, odom_frame;

std::unique_ptr<tf2_ros::Buffer> tf2_buffer;
std::unique_ptr<tf2_ros::TransformListener> tf2_listener;

int main(int argc, char ** argv)
{
  ros::init(argc, argv, "odometry_tf2_publisher");
  ROS_INFO("[OTFP] Starting OdometryTFPublisher");

  /* ros set up*/
  ros::NodeHandle private_nh_("~");
  ros::NodeHandle nh_;

  /* Link states topic has an absolute namespace because gazebo is always run at the top level */
  ros::Subscriber link_states_sub =
          nh_.subscribe<gazebo_msgs::LinkStates>("/gazebo/link_states", 10,
                  &link_states_callback,ros::TransportHints().tcpNoDelay().reliable());

  ros::Publisher odom_pub = nh_.advertise<nav_msgs::Odometry>("vio_node/odom_flu", 10);

  tf2_ros::TransformBroadcaster broadcaster;
  tf2_buffer = std::make_unique<tf2_ros::Buffer>(ros::Duration(1));
  tf2_listener =
          std::make_unique<tf2_ros::TransformListener>(*tf2_buffer, true);

  /* setup transforms */
  private_nh_.param<std::string>("map_frame", map_frame, "0/map");
  private_nh_.param<std::string>("body_frame", body_frame, "body");
  private_nh_.param<std::string>("gaz_body_link", gaz_body_link, "vision60::body");
  private_nh_.param<std::string>("odom_frame", odom_frame, body_frame);

  odom_trans.header.frame_id = map_frame;
  odom_trans.child_frame_id = odom_frame;

  odom_msg.header.frame_id = map_frame;
  odom_msg.child_frame_id = odom_frame;

  bool init = true;
  if (odom_frame != body_frame){
    init = false;
  }

  ros::Rate r(150);
  while (ros::ok()) {

    ros::spinOnce();

    if (!init){
      try {
        tf2::fromMsg(
                tf2_buffer->lookupTransform(odom_frame, body_frame, ros::Time::now(), ros::Duration(5)), body_to_pose);
      } catch (tf2::LookupException e) {
        ROS_ERROR("[PCLTRACKFUSE]::INIT No transform %s to %s: %s",
                  odom_frame.c_str(), body_frame.c_str(), e.what());
        continue;
      }
    }

    broadcaster.sendTransform(odom_trans);
    odom_pub.publish(odom_msg);
    r.sleep();
  }
  return 0;
}

void link_states_callback(gazebo_msgs::LinkStates ls_msg)
{
  odom_trans.header.stamp = ros::Time::now();
  odom_msg.header.stamp = ros::Time::now();

  int idx = 0;
  for (size_t i = 0; i < ls_msg.name.size(); i++) {
    if (ls_msg.name[i] == gaz_body_link) {
      idx = i;
      break;
    }
  }

  /* transform odometry to odom frame */
  if (odom_frame != body_frame){
    geometry_msgs::PoseStamped tmp_pose_stamped, transformed_pose_stamped;
    tmp_pose_stamped.pose = ls_msg.pose[idx];
    tmp_pose_stamped.header.frame_id = body_frame;
    tmp_pose_stamped.header.stamp = odom_trans.header.stamp;

    tf2_buffer->transform<geometry_msgs::PoseStamped>(tmp_pose_stamped, transformed_pose_stamped, odom_frame, ros::Duration(1));
//    ROS_WARN("Pose In: %f, %f, %f", tmp_pose_stamped.pose.position.x,
//             tmp_pose_stamped.pose.position.y,
//             tmp_pose_stamped.pose.position.z);
//    ROS_WARN("Transformed Pose: %f, %f, %f", transformed_pose_stamped.pose.position.x,
//             transformed_pose_stamped.pose.position.y,
//             transformed_pose_stamped.pose.position.z);

  }

  odom_trans.transform.translation.x = ls_msg.pose[idx].position.x;
  odom_trans.transform.translation.y = ls_msg.pose[idx].position.y;
  odom_trans.transform.translation.z = ls_msg.pose[idx].position.z;

  odom_trans.transform.rotation.x = ls_msg.pose[idx].orientation.x;
  odom_trans.transform.rotation.y = ls_msg.pose[idx].orientation.y;
  odom_trans.transform.rotation.z = ls_msg.pose[idx].orientation.z;
  odom_trans.transform.rotation.w = ls_msg.pose[idx].orientation.w;

  /* odom message */
  odom_msg.pose.pose.position.x = ls_msg.pose[idx].position.x;
  odom_msg.pose.pose.position.y = ls_msg.pose[idx].position.y;
  odom_msg.pose.pose.position.z = ls_msg.pose[idx].position.z;

  odom_msg.pose.pose.orientation.x = ls_msg.pose[idx].orientation.x;
  odom_msg.pose.pose.orientation.y = ls_msg.pose[idx].orientation.y;
  odom_msg.pose.pose.orientation.z = ls_msg.pose[idx].orientation.z;
  odom_msg.pose.pose.orientation.w = ls_msg.pose[idx].orientation.w;

  odom_msg.twist.twist.linear.x = ls_msg.twist[idx].linear.x;
  odom_msg.twist.twist.linear.y = ls_msg.twist[idx].linear.y;
  odom_msg.twist.twist.linear.z = ls_msg.twist[idx].linear.z;

  odom_msg.twist.twist.angular.x = ls_msg.twist[idx].angular.x;
  odom_msg.twist.twist.angular.y = ls_msg.twist[idx].angular.y;
  odom_msg.twist.twist.angular.z = ls_msg.twist[idx].angular.z;

}
