#ifndef AUTO_AIM__PLANNER_EXPLICIT_HPP
#define AUTO_AIM__PLANNER_EXPLICIT_HPP

#include <Eigen/Dense>
#include <optional>
#include <string>

#include "tasks/auto_aim/planner/planner.hpp"

namespace auto_aim
{

class PlannerExplicit
{
public:
  explicit PlannerExplicit(const std::string & config_path);

  Plan plan(Target target, double bullet_speed);
  Plan plan(std::optional<Target> target, double bullet_speed);

private:
  std::string config_path_;

  // Shared behavior params (aligned with Planner)
  double yaw_offset_;
  double pitch_offset_;
  double fire_thresh_;
  double low_speed_delay_time_, high_speed_delay_time_, decision_speed_;
  double max_yaw_acc_, max_pitch_acc_;

  // Explicit-search params
  double explicit_min_T_;
  double explicit_max_T_;
  double explicit_step_T_;

  Eigen::Vector4d debug_xyza;

  Eigen::Matrix<double, 2, 1> aim(const Target & target, double bullet_speed);

  Trajectory get_raw_trajectory(Target & target, double yaw0, double bullet_speed);
  Trajectory get_trajectory_explicit(Target & target, double yaw0, double bullet_speed);

  static int find_switch_index(const Eigen::VectorXd & angle_seq);

  static Eigen::VectorXd apply_quintic_transition(
    const Eigen::VectorXd & x, double v0, double v1, int i0, int i1, double dt);

  static double max_abs_acc(const Eigen::VectorXd & x, double dt);

  static void fill_velocity(Trajectory & traj);
};

}  // namespace auto_aim

#endif  // AUTO_AIM__PLANNER_EXPLICIT_HPP