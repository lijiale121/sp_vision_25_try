#include "planner_explicit.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "tools/math_tools.hpp"
#include "tools/trajectory.hpp"
#include "tools/yaml.hpp"

namespace auto_aim
{

PlannerExplicit::PlannerExplicit(const std::string & config_path) : config_path_(config_path)
{
  auto yaml = tools::load(config_path);
  yaw_offset_ = tools::read<double>(yaml, "yaw_offset") / 57.3;
  pitch_offset_ = tools::read<double>(yaml, "pitch_offset") / 57.3;
  fire_thresh_ = tools::read<double>(yaml, "fire_thresh");
  decision_speed_ = tools::read<double>(yaml, "decision_speed");
  high_speed_delay_time_ = tools::read<double>(yaml, "high_speed_delay_time");
  low_speed_delay_time_ = tools::read<double>(yaml, "low_speed_delay_time");

  max_yaw_acc_ = tools::read<double>(yaml, "max_yaw_acc");
  max_pitch_acc_ = tools::read<double>(yaml, "max_pitch_acc");

  // Optional explicit planner params, with safe defaults
  explicit_min_T_ = yaml["explicit_min_T"] ? tools::read<double>(yaml, "explicit_min_T") : 0.04;
  explicit_max_T_ = yaml["explicit_max_T"] ? tools::read<double>(yaml, "explicit_max_T") : 0.30;
  explicit_step_T_ = yaml["explicit_step_T"] ? tools::read<double>(yaml, "explicit_step_T") : 0.01;
}

Plan PlannerExplicit::plan(std::optional<Target> target, double bullet_speed)
{
  if (!target.has_value()) return {false};

  double delay_time =
    std::abs(target->ekf_x()[7]) > decision_speed_ ? high_speed_delay_time_ : low_speed_delay_time_;

  auto future = std::chrono::steady_clock::now() + std::chrono::microseconds(int(delay_time * 1e6));
  target->predict(future);

  return plan(*target, bullet_speed);
}

Plan PlannerExplicit::plan(Target target, double bullet_speed)
{
  if (bullet_speed < 10 || bullet_speed > 25) {
    bullet_speed = 22;
  }

  // Predict fly time
  Eigen::Vector3d xyz;
  double min_dist = 1e10;
  for (auto & xyza : target.armor_xyza_list()) {
    double dist = xyza.head<2>().norm();
    if (dist < min_dist) {
      min_dist = dist;
      xyz = xyza.head<3>();
    }
  }

  auto bullet_traj = tools::Trajectory(bullet_speed, min_dist, xyz.z());
  target.predict(bullet_traj.fly_time);

  double yaw0 = 0.0;
  Trajectory traj;
  try {
    yaw0 = aim(target, bullet_speed)(0);
    traj = get_trajectory_explicit(target, yaw0, bullet_speed);
  } catch (const std::exception &) {
    return {false};
  }

  // Explicit plan directly outputs reference trajectory point at HALF_HORIZON
  Plan out;
  out.control = true;

  out.target_yaw = tools::limit_rad(traj(0, HALF_HORIZON) + yaw0);
  out.target_pitch = traj(2, HALF_HORIZON);

  out.yaw = out.target_yaw;
  out.yaw_vel = traj(1, HALF_HORIZON);
  out.yaw_acc = (HALF_HORIZON > 0 && HALF_HORIZON < HORIZON - 1)
                  ? (traj(1, HALF_HORIZON + 1) - traj(1, HALF_HORIZON - 1)) / (2 * DT)
                  : 0.0f;

  out.pitch = out.target_pitch;
  out.pitch_vel = traj(3, HALF_HORIZON);
  out.pitch_acc = (HALF_HORIZON > 0 && HALF_HORIZON < HORIZON - 1)
                    ? (traj(3, HALF_HORIZON + 1) - traj(3, HALF_HORIZON - 1)) / (2 * DT)
                    : 0.0f;

  constexpr int shoot_offset = 2;
  out.fire =
    std::hypot(
      traj(0, HALF_HORIZON + shoot_offset) - traj(0, HALF_HORIZON + shoot_offset),
      traj(2, HALF_HORIZON + shoot_offset) - traj(2, HALF_HORIZON + shoot_offset)) < fire_thresh_;

  return out;
}

Eigen::Matrix<double, 2, 1> PlannerExplicit::aim(const Target & target, double bullet_speed)
{
  Eigen::Vector3d xyz;
  double yaw = 0.0;
  double min_dist = 1e10;

  for (auto & xyza : target.armor_xyza_list()) {
    double dist = xyza.head<2>().norm();
    if (dist < min_dist) {
      min_dist = dist;
      xyz = xyza.head<3>();
      yaw = xyza[3];
    }
  }
  debug_xyza = Eigen::Vector4d(xyz.x(), xyz.y(), xyz.z(), yaw);

  auto azim = std::atan2(xyz.y(), xyz.x());
  auto bullet_traj = tools::Trajectory(bullet_speed, min_dist, xyz.z());
  if (bullet_traj.unsolvable) throw std::runtime_error("Unsolvable bullet trajectory!");

  return {tools::limit_rad(azim + yaw_offset_), -bullet_traj.pitch - pitch_offset_};
}

Trajectory PlannerExplicit::get_raw_trajectory(Target & target, double yaw0, double bullet_speed)
{
  Trajectory traj;
  target.predict(-DT * (HALF_HORIZON + 1));
  auto yaw_pitch_last = aim(target, bullet_speed);

  target.predict(DT);
  auto yaw_pitch = aim(target, bullet_speed);

  for (int i = 0; i < HORIZON; i++) {
    target.predict(DT);
    auto yaw_pitch_next = aim(target, bullet_speed);

    auto yaw_vel = tools::limit_rad(yaw_pitch_next(0) - yaw_pitch_last(0)) / (2 * DT);
    auto pitch_vel = (yaw_pitch_next(1) - yaw_pitch_last(1)) / (2 * DT);

    traj.col(i) << tools::limit_rad(yaw_pitch(0) - yaw0), yaw_vel, yaw_pitch(1), pitch_vel;

    yaw_pitch_last = yaw_pitch;
    yaw_pitch = yaw_pitch_next;
  }

  return traj;
}

Trajectory PlannerExplicit::get_trajectory_explicit(Target & target, double yaw0, double bullet_speed)
{
  Trajectory raw = get_raw_trajectory(target, yaw0, bullet_speed);

  Eigen::VectorXd yaw = raw.row(0).transpose();
  Eigen::VectorXd pitch = raw.row(2).transpose();

  int sw = find_switch_index(yaw);
  if (sw < 0) return raw;

  Eigen::VectorXd best_yaw = yaw;
  Eigen::VectorXd best_pitch = pitch;
  bool found = false;

  for (double T = explicit_min_T_; T <= explicit_max_T_ + 1e-9; T += explicit_step_T_) {
    int n = std::max(2, int(std::round(T / DT)));
    int i0 = std::max(0, sw - n / 2);
    int i1 = std::min(HORIZON - 1, i0 + n);
    if (i1 - i0 < 2) continue;

    double vy0 = (i0 > 0) ? tools::limit_rad(yaw(i0) - yaw(i0 - 1)) / DT : 0.0;
    double vy1 = (i1 < HORIZON - 1) ? tools::limit_rad(yaw(i1 + 1) - yaw(i1)) / DT : 0.0;
    double vp0 = (i0 > 0) ? (pitch(i0) - pitch(i0 - 1)) / DT : 0.0;
    double vp1 = (i1 < HORIZON - 1) ? (pitch(i1 + 1) - pitch(i1)) / DT : 0.0;

    auto y2 = apply_quintic_transition(yaw, vy0, vy1, i0, i1, DT);
    auto p2 = apply_quintic_transition(pitch, vp0, vp1, i0, i1, DT);

    if (max_abs_acc(y2, DT) <= max_yaw_acc_ && max_abs_acc(p2, DT) <= max_pitch_acc_) {
      best_yaw = y2;
      best_pitch = p2;
      found = true;
      break;  // shortest feasible transition
    }
  }

  if (!found) {
    int n = std::max(2, int(std::round(explicit_max_T_ / DT)));
    int i0 = std::max(0, sw - n / 2);
    int i1 = std::min(HORIZON - 1, i0 + n);
    best_yaw = apply_quintic_transition(yaw, 0.0, 0.0, i0, i1, DT);
    best_pitch = apply_quintic_transition(pitch, 0.0, 0.0, i0, i1, DT);
  }

  Trajectory out = raw;
  out.row(0) = best_yaw.transpose();
  out.row(2) = best_pitch.transpose();
  fill_velocity(out);
  return out;
}

int PlannerExplicit::find_switch_index(const Eigen::VectorXd & angle_seq)
{
  int idx = -1;
  double best = 0.0;

  for (int i = 2; i < angle_seq.size() - 2; ++i) {
    double d0 = tools::limit_rad(angle_seq(i) - angle_seq(i - 1));
    double d1 = tools::limit_rad(angle_seq(i + 1) - angle_seq(i));
    double jump = std::abs(d1 - d0);

    if (jump > best) {
      best = jump;
      idx = i;
    }
  }

  return idx;
}

Eigen::VectorXd PlannerExplicit::apply_quintic_transition(
  const Eigen::VectorXd & x, double v0, double v1, int i0, int i1, double dt)
{
  Eigen::VectorXd y = x;
  if (i1 <= i0) return y;

  double T = (i1 - i0) * dt;
  if (T <= 1e-9) return y;

  const double x0 = x(i0);
  const double x1 = x(i1);

  Eigen::Matrix<double, 6, 6> A;
  Eigen::Matrix<double, 6, 1> b;

  A << 1, 0, 0, 0, 0, 0,
       0, 1, 0, 0, 0, 0,
       0, 0, 2, 0, 0, 0,
       1, T, T * T, T * T * T, T * T * T * T, T * T * T * T * T,
       0, 1, 2 * T, 3 * T * T, 4 * T * T * T, 5 * T * T * T * T,
       0, 0, 2, 6 * T, 12 * T * T, 20 * T * T * T;

  b << x0, v0, 0.0, x1, v1, 0.0;

  Eigen::Matrix<double, 6, 1> c = A.colPivHouseholderQr().solve(b);

  for (int k = i0; k <= i1; ++k) {
    double t = (k - i0) * dt;
    double t2 = t * t;
    double t3 = t2 * t;
    double t4 = t3 * t;
    double t5 = t4 * t;
    y(k) = c(0) + c(1) * t + c(2) * t2 + c(3) * t3 + c(4) * t4 + c(5) * t5;
  }

  return y;
}

double PlannerExplicit::max_abs_acc(const Eigen::VectorXd & x, double dt)
{
  double peak = 0.0;
  for (int i = 1; i < x.size() - 1; ++i) {
    double a = (x(i + 1) - 2.0 * x(i) + x(i - 1)) / (dt * dt);
    peak = std::max(peak, std::abs(a));
  }
  return peak;
}

void PlannerExplicit::fill_velocity(Trajectory & traj)
{
  for (int i = 0; i < HORIZON; ++i) {
    double vy = 0.0;
    double vp = 0.0;

    if (i > 0 && i < HORIZON - 1) {
      vy = tools::limit_rad(traj(0, i + 1) - traj(0, i - 1)) / (2 * DT);
      vp = (traj(2, i + 1) - traj(2, i - 1)) / (2 * DT);
    } else if (i > 0) {
      vy = tools::limit_rad(traj(0, i) - traj(0, i - 1)) / DT;
      vp = (traj(2, i) - traj(2, i - 1)) / DT;
    }

    traj(1, i) = vy;
    traj(3, i) = vp;
  }
}

}  // namespace auto_aim