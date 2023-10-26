#include <gtest/gtest.h>

#include "utils.h"

TEST(Utils, Jl_Jl_inv) {
  Vector3d v{1, 2, 3};

  auto jl_inv_mat = Jl_inv(v);
  auto jl         = Jl(v);

  EXPECT_TRUE(jl_inv_mat.isApprox(jl.inverse()));
}

TEST(Utils, Jl_Jr) {
  Vector3d v{1, 2, 3};

  auto jl = Jl(v);
  auto jr = Jr(-v);

  EXPECT_TRUE(jl.isApprox(jr));
}
