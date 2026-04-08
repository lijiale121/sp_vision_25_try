#include <fmt/core.h>

#include <chrono>
#include <fstream>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "tasks/auto_aim/planner/planner_explicit.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"

const std::string keys =
  "{help h usage ? |                   | 输出命令行参数说明 }"
  "{config-path c  | configs/demo.yaml | yaml配置文件的路径}"
  "{bullet-speed b | 27                | 子弹速度(m/s)      }"
  "{start-index s  | 0                 | 视频起始帧下标      }"
  "{end-index e    | 0                 | 视频结束帧下标      }"
  "{@input-path    | assets/demo/demo  | avi和txt文件的路径 }";

int main(int argc, char * argv[])
{
  auto get_arg_value = [&](const std::string & short_key, const std::string & long_key) -> std::string {
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == short_key || arg == long_key) {
        if (i + 1 < argc) return argv[i + 1];
      }
      if (arg.rfind(short_key + "=", 0) == 0) return arg.substr(short_key.size() + 1);
      if (arg.rfind(long_key + "=", 0) == 0) return arg.substr(long_key.size() + 1);
    }
    return "";
  };

  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }

  auto input_path = cli.get<std::string>(0);
  auto config_path = cli.get<std::string>("config-path");
  auto bullet_speed = cli.get<double>("bullet-speed");
  auto start_index = cli.get<int>("start-index");
  auto end_index = cli.get<int>("end-index");

  // Work around OpenCV parser corner case: "-c path" may parse as "true".
  if (config_path == "true") {
    auto fallback = get_arg_value("-c", "--config-path");
    if (!fallback.empty()) config_path = fallback;
  }

  tools::Plotter plotter;
  tools::Exiter exiter;

  auto video_path = fmt::format("{}.avi", input_path);
  auto text_path = fmt::format("{}.txt", input_path);
  cv::VideoCapture video(video_path);
  std::ifstream text(text_path);

  auto_aim::YOLO yolo(config_path);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::PlannerExplicit planner(config_path);

  if (!video.isOpened()) {
    tools::logger()->error("Cannot open video: {}", video_path);
    return -1;
  }
  if (!text.is_open()) {
    tools::logger()->error("Cannot open pose file: {}", text_path);
    return -1;
  }

  cv::Mat img;
  auto t0 = std::chrono::steady_clock::now();
  video.set(cv::CAP_PROP_POS_FRAMES, start_index);

  for (int i = 0; i < start_index; i++) {
    double t, w, x, y, z;
    text >> t >> w >> x >> y >> z;
  }

  for (int frame_count = start_index; !exiter.exit(); frame_count++) {
    if (end_index > 0 && frame_count > end_index) break;

    video.read(img);
    if (img.empty()) break;

    double t, w, x, y, z;
    text >> t >> w >> x >> y >> z;
    auto timestamp = t0 + std::chrono::microseconds(int(t * 1e6));

    solver.set_R_gimbal2world({w, x, y, z});

    auto yolo_start = std::chrono::steady_clock::now();
    auto armors = yolo.detect(img, frame_count);
    auto tracker_start = std::chrono::steady_clock::now();
    auto targets = tracker.track(armors, timestamp);
    auto planner_start = std::chrono::steady_clock::now();

    auto plan = planner.plan(targets.empty() ? std::nullopt : std::optional(targets.front()), bullet_speed);

    auto finish = std::chrono::steady_clock::now();
    tools::logger()->info(
      "[{}] yolo: {:.1f}ms, tracker: {:.1f}ms, planner_explicit: {:.1f}ms", frame_count,
      tools::delta_time(tracker_start, yolo_start) * 1e3,
      tools::delta_time(planner_start, tracker_start) * 1e3,
      tools::delta_time(finish, planner_start) * 1e3);

    tools::draw_text(
      img,
      fmt::format(
        "plan control:{} fire:{} yaw:{:.2f} pitch:{:.2f}", plan.control, plan.fire,
        plan.yaw * 57.3, plan.pitch * 57.3),
      {10, 60}, {154, 50, 205});

    Eigen::Quaternion q{w, x, y, z};
    auto gimbal_yaw = tools::eulers(q, 2, 1, 0)[0];
    tools::draw_text(img, fmt::format("gimbal yaw:{:.2f}", gimbal_yaw * 57.3), {10, 90}, {255, 255, 255});

    if (!targets.empty()) {
      auto target = targets.front();
      auto armor_xyza_list = target.armor_xyza_list();
      for (const auto & xyza : armor_xyza_list) {
        auto image_points = solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
        tools::draw_points(img, image_points, {0, 255, 0});
      }
    }

    nlohmann::json data;
    data["armor_num"] = armors.size();
    data["gimbal_yaw"] = gimbal_yaw * 57.3;
    data["plan_control"] = plan.control;
    data["plan_fire"] = plan.fire;
    data["plan_yaw"] = plan.yaw * 57.3;
    data["plan_yaw_vel"] = plan.yaw_vel * 57.3;
    data["plan_yaw_acc"] = plan.yaw_acc * 57.3;
    data["plan_pitch"] = plan.pitch * 57.3;
    data["plan_pitch_vel"] = plan.pitch_vel * 57.3;
    data["plan_pitch_acc"] = plan.pitch_acc * 57.3;
    plotter.plot(data);

    cv::resize(img, img, {}, 0.5, 0.5);
    cv::imshow("planner_explicit_video_test", img);
    auto key = cv::waitKey(30);
    if (key == 'q') break;
  }

  return 0;
}
