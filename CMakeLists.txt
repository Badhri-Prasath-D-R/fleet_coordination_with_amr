cmake_minimum_required(VERSION 3.5)
project(bcr_bot)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

install(DIRECTORY launch meshes models urdf worlds rviz
    DESTINATION share/${PROJECT_NAME}/
)

install( DIRECTORY config/
         DESTINATION share/${PROJECT_NAME}/config)


# The fleet coordination logic lives in an importable Python package rather
# than inside the node scripts, so the geometry, planner, arbitration and
# auction code can be unit-tested and replayed against a log without ROS.
ament_python_install_package(fleet_coordination)

install(PROGRAMS
        scripts/remapper.py
        scripts/fleet_collision_node.py
        scripts/fleet_navigator.py
        scripts/fleet_task_allocator.py
        scripts/fleet_task_dispatcher.py
        scripts/fleet_monitor.py
        scripts/replay_proximity_log.py
        scripts/verify_fleet_config.py
        DESTINATION lib/${PROJECT_NAME})

ament_package()
