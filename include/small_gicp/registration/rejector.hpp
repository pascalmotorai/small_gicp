// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace small_gicp {

/// @brief Null correspondence rejector. This class accepts all input correspondences.
struct NullRejector {
  template <typename TargetPointCloud, typename SourcePointCloud>
  bool operator()(const TargetPointCloud& target, const SourcePointCloud& source, const Eigen::Isometry3d& T, size_t target_index, size_t source_index, double sq_dist) const {
    return false;
  }
};

/// @brief Rejecting correspondences with large distances.
struct DistanceRejector {
  DistanceRejector() : max_dist_sq(1.0) {}

  template <typename TargetPointCloud, typename SourcePointCloud>
  bool operator()(const TargetPointCloud& target, const SourcePointCloud& source, const Eigen::Isometry3d& T, size_t target_index, size_t source_index, double sq_dist) const {
    return sq_dist > max_dist_sq;
  }

  double max_dist_sq;  ///< Maximum squared distance between corresponding points
};


struct LabelDistanceRejector
{
    double max_dist_sq;

    LabelDistanceRejector() : max_dist_sq(0.7 * 0.7) {} // Default threshold: 0.7m
    LabelDistanceRejector(double max_dist) : max_dist_sq(max_dist * max_dist) {}

    template <typename TargetPointCloud, typename SourcePointCloud>
    bool operator()(const TargetPointCloud &target, const SourcePointCloud &source,
                    const Eigen::Isometry3d &T, size_t target_index,
                    size_t source_index, double sq_dist) const
    {
        if (target.points[target_index].z() != source.points[source_index].z())
        {
            return true; // reject label mismatch
        }

        if (sq_dist > max_dist_sq)
        {
            return true; // reject if distance exceeds max_dist_sq
        }

        return false; // accept otherwise
    }
};

}  // namespace small_gicp
