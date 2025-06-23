#pragma once

#include <memory>
#include <vector>
#include <Eigen/Core>
#include <small_gicp/points/traits.hpp>
#include <small_gicp/factors/icp_factor.hpp>

namespace small_gicp {

struct LabeledPointCloud {
public:
  using Ptr = std::shared_ptr<LabeledPointCloud>;
  using ConstPtr = std::shared_ptr<const LabeledPointCloud>;

  LabeledPointCloud() {}

  template <typename T, int D, typename Allocator1, typename Allocator2>
  LabeledPointCloud(const std::vector<Eigen::Matrix<T, D, 1>, Allocator1>& points,
                    const std::vector<int, Allocator2>& labels) {
    this->resize(points.size());
    for (size_t i = 0; i < points.size(); i++) {
      this->point(i) << points[i].template cast<double>().template head<3>(), 1.0;
      this->label(i) = (i < labels.size()) ? labels[i] : 0;
    }
  }

  ~LabeledPointCloud() {}

  size_t size() const { return points.size(); }
  bool empty() const { return points.empty(); }

  void resize(size_t n) {
    points.resize(n);
    normals.resize(n);
    covs.resize(n);
    labels.resize(n);
  }

  Eigen::Vector4d& point(size_t i) { return points[i]; }
  Eigen::Vector4d& normal(size_t i) { return normals[i]; }
  Eigen::Matrix4d& cov(size_t i) { return covs[i]; }
  int& label(size_t i) { return labels[i]; }

  const Eigen::Vector4d& point(size_t i) const { return points[i]; }
  const Eigen::Vector4d& normal(size_t i) const { return normals[i]; }
  const Eigen::Matrix4d& cov(size_t i) const { return covs[i]; }
  const int& label(size_t i) const { return labels[i]; }

public:
  std::vector<Eigen::Vector4d> points;
  std::vector<Eigen::Vector4d> normals;
  std::vector<Eigen::Matrix4d> covs;
  std::vector<int> labels;
};

namespace traits {

template <>
struct Traits<LabeledPointCloud> {
  using Points = LabeledPointCloud;

  static size_t size(const Points& points) { return points.size(); }

  static bool has_points(const Points& points) { return !points.points.empty(); }
  static bool has_normals(const Points& points) { return !points.normals.empty(); }
  static bool has_covs(const Points& points) { return !points.covs.empty(); }
  static bool has_labels(const Points& points) { return !points.labels.empty(); }

  static const Eigen::Vector4d& point(const Points& points, size_t i) { return points.point(i); }
  static const Eigen::Vector4d& normal(const Points& points, size_t i) { return points.normal(i); }
  static const Eigen::Matrix4d& cov(const Points& points, size_t i) { return points.cov(i); }
  static int label(const Points& points, size_t i) { return points.label(i); }

  static void resize(Points& points, size_t n) { points.resize(n); }
  static void set_point(Points& points, size_t i, const Eigen::Vector4d& pt) { points.point(i) = pt; }
  static void set_normal(Points& points, size_t i, const Eigen::Vector4d& n) { points.normal(i) = n; }
  static void set_cov(Points& points, size_t i, const Eigen::Matrix4d& cov) { points.cov(i) = cov; }
  static void set_label(Points& points, size_t i, int l) { points.label(i) = l; }
};

}  // namespace traits

}  // namespace small_gicp
