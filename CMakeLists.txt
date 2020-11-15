cmake_minimum_required(VERSION 3.16)
project(sunfish)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

if (APPLE)
	include_directories(/usr/local/include)
	set(STD_CXX_FS c++fs)
endif()

include_directories(hatchetfish/include)
include_directories(fluxions/include)
include_directories(fluxions-gte/include)
include_directories(fluxions-base/include)
include_directories(include)

find_package(GLEW REQUIRED)
target_link_libraries(main PRIVATE GLEW::GLEW)

add_executable(${PROJECT_NAME}
	sunfish.cpp
	)


