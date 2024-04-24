# Setup
import os
import re

import symforce

symforce.set_symbolic_api("symengine")
symforce.set_log_level("warning")

# Set epsilon to a symbol for safe code generation.  For more information, see the Epsilon tutorial:
# https://symforce.org/tutorials/epsilon_tutorial.html
symforce.set_epsilon_to_symbol()

import symforce.symbolic as sf
from symforce import codegen
from symforce.codegen import codegen_util
from symforce.notebook_util import display
from symforce.notebook_util import display_code_file
from symforce.values import Values


path_output = os.path.abspath(os.path.dirname(__file__)) + "/../gen"


class Vector6(sf.Matrix):
    SHAPE = (6, 1)


######################################################
#   generate code for surfel_binary_match_4_samples
######################################################
def surfel_unary_match(
    sp2l: Vector6,
    sp2r: Vector6,
    factor_sp2: sf.Scalar,
    normal: sf.V3,
    center_sp1: sf.V3,
    center_sp2: sf.V3,
    epsilon: sf.Scalar = sf.epsilon(),
) -> sf.Vector1:
    sp2l_rot = sf.Rot3.from_tangent(sp2l[0:3])
    sp2r_rot = sf.Rot3.from_tangent(sp2r[0:3])
    sp2_pose_cor = sf.Pose3(
        sp2l_rot
        * sf.Rot3.from_tangent(
            factor_sp2
            * sf.V3.from_flat_list((sp2l_rot.inverse() * sp2r_rot).to_tangent())
        ),
        (1.0 - factor_sp2) * sp2l[3:6] + factor_sp2 * sp2r[3:6],
    )
    return normal.dot(center_sp1 - sp2_pose_cor * center_sp2)


az_el_codegen = codegen.Codegen.function(
    func=surfel_unary_match,
    config=codegen.CppConfig(),
)

codegen_with_jacobians = az_el_codegen.with_jacobians(
    which_args=["sp2l", "sp2r"],
    include_results=True,
)

data = codegen_with_jacobians.generate_function(output_dir=path_output)


######################################################
#   generate code for surfel_binary_match_4_samples
######################################################
def surfel_binary_match_4_samples(
    sp1l: Vector6,
    sp1r: Vector6,
    sp2l: Vector6,
    sp2r: Vector6,
    factor_sp1: sf.Scalar,
    factor_sp2: sf.Scalar,
    normal: sf.V3,
    center_sp1: sf.V3,
    center_sp2: sf.V3,
    epsilon: sf.Scalar = sf.epsilon(),
) -> sf.Vector1:
    sp1l_rot = sf.Rot3.from_tangent(sp1l[0:3])
    sp1r_rot = sf.Rot3.from_tangent(sp1r[0:3])
    sp1_pose_cor = sf.Pose3(
        sp1l_rot
        * sf.Rot3.from_tangent(
            factor_sp1
            * sf.V3.from_flat_list((sp1l_rot.inverse() * sp1r_rot).to_tangent())
        ),
        (1.0 - factor_sp1) * sp1l[3:6] + factor_sp1 * sp1r[3:6],
    )
    sp2l_rot = sf.Rot3.from_tangent(sp2l[0:3])
    sp2r_rot = sf.Rot3.from_tangent(sp2r[0:3])
    sp2_pose_cor = sf.Pose3(
        sp2l_rot
        * sf.Rot3.from_tangent(
            factor_sp2
            * sf.V3.from_flat_list((sp2l_rot.inverse() * sp2r_rot).to_tangent())
        ),
        (1.0 - factor_sp2) * sp2l[3:6] + factor_sp2 * sp2r[3:6],
    )
    return normal.dot(sp1_pose_cor * center_sp1 - sp2_pose_cor * center_sp2)


az_el_codegen = codegen.Codegen.function(
    func=surfel_binary_match_4_samples,
    config=codegen.CppConfig(),
)

codegen_with_jacobians = az_el_codegen.with_jacobians(
    which_args=["sp1l", "sp1r", "sp2l", "sp2r"],
    include_results=True,
)

data = codegen_with_jacobians.generate_function(output_dir=path_output)


######################################################
#   generate code for surfel_binary_match_3_samples
######################################################
def surfel_binary_match_3_samples(
    sp1l: Vector6,
    sp1r: Vector6,
    sp2r: Vector6,
    factor_sp1: sf.Scalar,
    factor_sp2: sf.Scalar,
    normal: sf.V3,
    center_sp1: sf.V3,
    center_sp2: sf.V3,
    epsilon: sf.Scalar = sf.epsilon(),
) -> sf.Vector1:
    return surfel_binary_match_4_samples(
        sp1l,
        sp1r,
        sp1r,
        sp2r,
        factor_sp1,
        factor_sp2,
        normal,
        center_sp1,
        center_sp2,
        epsilon,
    )


az_el_codegen = codegen.Codegen.function(
    func=surfel_binary_match_3_samples,
    config=codegen.CppConfig(),
)

codegen_with_jacobians = az_el_codegen.with_jacobians(
    # Just compute wrt the pose and point, not epsilon
    which_args=["sp1l", "sp1r", "sp2r"],
    # Include value, not just jacobians
    include_results=True,
)

data = codegen_with_jacobians.generate_function(output_dir=path_output)


######################################################
#   generate code for surfel_binary_match_2_samples
######################################################
def surfel_binary_match_2_samples(
    sp1l: Vector6,
    sp1r: Vector6,
    factor_sp1: sf.Scalar,
    factor_sp2: sf.Scalar,
    normal: sf.V3,
    center_sp1: sf.V3,
    center_sp2: sf.V3,
    epsilon: sf.Scalar = sf.epsilon(),
) -> sf.Vector1:
    return surfel_binary_match_4_samples(
        sp1l,
        sp1r,
        sp1l,
        sp1r,
        factor_sp1,
        factor_sp2,
        normal,
        center_sp1,
        center_sp2,
        epsilon,
    )


az_el_codegen = codegen.Codegen.function(
    func=surfel_binary_match_2_samples,
    config=codegen.CppConfig(),
)

codegen_with_jacobians = az_el_codegen.with_jacobians(
    # Just compute wrt the pose and point, not epsilon
    which_args=["sp1l", "sp1r"],
    # Include value, not just jacobians
    include_results=True,
)

data = codegen_with_jacobians.generate_function(output_dir=path_output)


for root, dirs, files in os.walk(path_output):
    for file_name in files:
        file_path = os.path.join(root, file_name)
        with open(file_path, "r", encoding="utf-8") as file:
            file_contents = file.read()

        new_contents = file_contents

        # update type of sample states
        new_contents = re.sub(
            r"const Eigen::Matrix<Scalar, 6, 1>",
            "Eigen::Map<const Eigen::Matrix<Scalar, 6, 1>>",
            new_contents,
        )

        # update type of jacobians
        new_contents = re.sub(
            r"Eigen::Matrix<([^,]+), 1, 6>\* const (\w+) = nullptr",
            r"\1* \2 = nullptr",
            new_contents,
        )
        new_contents = re.sub(
            r"(Eigen::Matrix<([^,]+), 1, 6>&) (\w+) = \(\*(\w+)\);",
            r"Eigen::Map<Eigen::Matrix<\2, 1, 6>> \3{\4};",
            new_contents,
        )

        if new_contents != file_contents:
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(new_contents)

            print(f"Updated file: {file_path}")
