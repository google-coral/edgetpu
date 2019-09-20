#include "src/cpp/basic/edgetpu_resource_manager.h"

#include <chrono>  // NOLINT
#include <random>
#include <thread>  // NOLINT

#include "absl/flags/parse.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

namespace coral {
namespace {

using EdgeTpuState = EdgeTpuResourceManager::EdgeTpuState;

class EdgeTpuResourceManagerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    resource_manager_ = EdgeTpuResourceManager::GetSingleton();
    ASSERT_TRUE(resource_manager_);
    unassigned_devices_ =
        resource_manager_->ListEdgeTpuPaths(EdgeTpuState::kUnassigned);
    ASSERT_GE(unassigned_devices_.size(), 1);
  }

  void TearDown() override {
    EXPECT_EQ(
        0, resource_manager_->ListEdgeTpuPaths(EdgeTpuState::kAssigned).size());
  }
  EdgeTpuResourceManager* resource_manager_;
  std::vector<std::string> unassigned_devices_;
};

TEST_F(EdgeTpuResourceManagerTest, GetAllEdgeTpuOnce) {
  std::vector<std::unique_ptr<EdgeTpuResource>> edgetpu_resources(
      unassigned_devices_.size());
  for (int i = 0; i < unassigned_devices_.size(); ++i) {
    EXPECT_EQ(
        i, resource_manager_->ListEdgeTpuPaths(EdgeTpuState::kAssigned).size());
    ASSERT_EQ(kEdgeTpuApiOk,
              resource_manager_->GetEdgeTpuResource(&edgetpu_resources[i]));
  }
}

TEST_F(EdgeTpuResourceManagerTest, GetSameTpuContextRepeatedly) {
  std::unique_ptr<EdgeTpuResource> edgetpu_once;
  EXPECT_EQ(kEdgeTpuApiOk, resource_manager_->GetEdgeTpuResource(
                               unassigned_devices_[0], &edgetpu_once));
  std::unique_ptr<EdgeTpuResource> edgetpu_twice;
  ASSERT_TRUE(edgetpu_once != nullptr);
  EXPECT_EQ(kEdgeTpuApiOk, resource_manager_->GetEdgeTpuResource(
                               unassigned_devices_[0], &edgetpu_twice));
  ASSERT_TRUE(edgetpu_twice != nullptr);
  EXPECT_EQ(edgetpu_once->path(), edgetpu_twice->path());
  EXPECT_EQ(edgetpu_once->context(), edgetpu_twice->context());
  std::unique_ptr<EdgeTpuResource> edgetpu_thrice;
  EXPECT_EQ(kEdgeTpuApiOk, resource_manager_->GetEdgeTpuResource(
                               unassigned_devices_[0], &edgetpu_thrice));
  ASSERT_TRUE(edgetpu_thrice != nullptr);
  EXPECT_EQ(edgetpu_once->path(), edgetpu_thrice->path());
  EXPECT_EQ(edgetpu_once->context(), edgetpu_thrice->context());
}

TEST_F(EdgeTpuResourceManagerTest, CheckDevicePath) {
  std::unique_ptr<EdgeTpuResource> edgetpu_resource;
  EXPECT_EQ(kEdgeTpuApiOk, resource_manager_->GetEdgeTpuResource(
                               unassigned_devices_[0], &edgetpu_resource));
  EXPECT_EQ(unassigned_devices_[0], edgetpu_resource->path());
}

TEST_F(EdgeTpuResourceManagerTest, DeviceNotExistError) {
  std::unique_ptr<EdgeTpuResource> edgetpu_resource;
  EXPECT_EQ(kEdgeTpuApiError, resource_manager_->GetEdgeTpuResource(
                                  "invalid_path", &edgetpu_resource));
  EXPECT_EQ("Path invalid_path does not map to an Edge TPU device.",
            resource_manager_->get_error_message());
}

TEST_F(EdgeTpuResourceManagerTest, ReclaimUnassignedDeviceError) {
  EXPECT_EQ(kEdgeTpuApiError,
            resource_manager_->ReclaimEdgeTpuResource("unassigned_device"));
  EXPECT_EQ("Trying to reclaim unassigned device: unassigned_device.",
            resource_manager_->get_error_message());
}

TEST_F(EdgeTpuResourceManagerTest, ExhaustAllEdgeTpu) {
  // No need to run this test if there's only one Edge TPU detected.
  if (unassigned_devices_.size() <= 1) {
    return;
  }

  // Exhaust all Edge TPU.
  std::vector<std::unique_ptr<EdgeTpuResource>> edgetpu_resources(
      unassigned_devices_.size());
  for (int i = 0; i < edgetpu_resources.size(); ++i) {
    EXPECT_EQ(kEdgeTpuApiOk,
              resource_manager_->GetEdgeTpuResource(&edgetpu_resources[i]));
    VLOG(1) << "assigned: " << edgetpu_resources[i]->path();
  }

  // Request one more Edge TPU to trigger the error.
  const std::string expected_error_message =
      "Multiple Edge TPUs detected and all have been mapped to at least one "
      "model. If you want to share one Edge TPU with multiple models, specify "
      "`device_path` name.";
  std::unique_ptr<EdgeTpuResource> another_resource;
  EXPECT_EQ(kEdgeTpuApiError,
            resource_manager_->GetEdgeTpuResource(&another_resource));
  EXPECT_EQ(expected_error_message, resource_manager_->get_error_message());
}

TEST_F(EdgeTpuResourceManagerTest, MultithreadTest) {
  const int num_devices = unassigned_devices_.size();
  const int num_threads = 3 * num_devices;

  // Each thread is randomly assigned to use a device.
  std::mt19937 generator(123456);
  auto get_device_assignments = [&generator, &num_devices,
                                 &num_threads]() -> std::vector<int> {
    std::vector<int> result(num_threads);
    std::uniform_int_distribution<> dis(0, num_devices - 1);
    for (int i = 0; i < num_threads; ++i) {
      result[i] = dis(generator);
    }
    return result;
  };
  std::vector<int> device_assignments = get_device_assignments();

  // Each thread will randomly sleep 100ms ~ 500ms then release the device.
  auto get_sleep_times = [&generator, &num_threads]() -> std::vector<int> {
    std::vector<int> result(num_threads);
    std::uniform_int_distribution<> dis(100, 500);
    for (int i = 0; i < num_threads; ++i) {
      result[i] = dis(generator);
    }
    return result;
  };
  std::vector<int> sleep_times = get_sleep_times();

  auto thread_job = [this](const std::string& device_path, int sleep_time) {
    std::unique_ptr<EdgeTpuResource> edgetpu_resource;
    EXPECT_EQ(kEdgeTpuApiOk, resource_manager_->GetEdgeTpuResource(
                                 device_path, &edgetpu_resource));
    EXPECT_EQ(device_path, edgetpu_resource->path());
    EXPECT_TRUE(edgetpu_resource->context());
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
  };

  std::vector<std::thread> workers(num_threads);
  for (int i = 0; i < num_threads; ++i) {
    const auto& device_path = unassigned_devices_[device_assignments[i]];
    workers[i] = std::thread(thread_job, device_path, sleep_times[i]);
  }

  for (int i = 0; i < num_threads; ++i) {
    workers[i].join();
  }
}

}  // namespace
}  // namespace coral

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  absl::ParseCommandLine(argc, argv);
  return RUN_ALL_TESTS();
}
