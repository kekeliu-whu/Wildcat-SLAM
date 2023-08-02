function(add_test_library_srcs ARG_SRC)
    set(TEST_LIB project_test_library)
    add_library(${TEST_LIB} ${ARG_SRC})
    target_link_libraries(${TEST_LIB} gtest)
endfunction()

function(google_test NAME ARG_SRC)
    add_executable(${NAME} ${ARG_SRC})

    # Make sure that gmock always includes the correct gtest/gtest.h.
    target_include_directories("${NAME}" SYSTEM PRIVATE
            "${GMOCK_INCLUDE_DIRS}")
    # todo use GMOCK_LIBRARIES, not gmock_main
    target_link_libraries("${NAME}" PUBLIC gmock_main)

    if (CATKIN_ENABLE_TESTING)
        add_test(${NAME} ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_BIN_DESTINATION}/${NAME})
        # add_test(${NAME} ${NAME} WORKING_DIRECTORY ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_BIN_DESTINATION})
        # get_test_property(${PROJECT_NAME} WORKING_DIRECTORY test_dir)
        # message("My test's working directory: ${test_dir}")
    else()
        add_test(${NAME} ${NAME})
    endif()
endfunction()

function(enable_automatic_test_and_benchmark)
    google_enable_testing()

    file(GLOB_RECURSE ALL_TESTS "*_test.cc")
    foreach (ABS_FIL ${ALL_TESTS})
        file(RELATIVE_PATH REL_FIL ${PROJECT_SOURCE_DIR} ${ABS_FIL})
        get_filename_component(DIR ${REL_FIL} DIRECTORY)
        get_filename_component(FIL_WE ${REL_FIL} NAME_WE)
        # Replace slashes as required for CMP0037.
        string(REPLACE "/" "." TEST_TARGET_NAME "${DIR}/${FIL_WE}")
        google_test("${TEST_TARGET_NAME}" ${ABS_FIL})
        target_link_libraries("${TEST_TARGET_NAME}" PUBLIC ${TEST_LIB} glog)
    endforeach ()

    file(GLOB_RECURSE ALL_BENCHMARKS "*_benchmark.cc")
    foreach (ABS_FIL ${ALL_BENCHMARKS})
        file(RELATIVE_PATH REL_FIL ${PROJECT_SOURCE_DIR} ${ABS_FIL})
        get_filename_component(DIR ${REL_FIL} DIRECTORY)
        get_filename_component(FIL_WE ${REL_FIL} NAME_WE)
        # Replace slashes as required for CMP0037.
        string(REPLACE "/" "." TEST_TARGET_NAME "${DIR}/${FIL_WE}")
        google_test("${TEST_TARGET_NAME}" ${ABS_FIL})
        target_link_libraries("${TEST_TARGET_NAME}" PUBLIC ${TEST_LIB} glog benchmark)
    endforeach ()
endfunction()

macro(google_enable_testing)
    enable_testing()
    #    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    #            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)
    #    find_package(GMock REQUIRED)
endmacro()
