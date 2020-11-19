cmake_minimum_required(VERSION 3.16)
project(sunfish)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

if (APPLE)
	include_directories(/usr/local/include)
	set(STD_CXX_FS c++fs)
endif()

# We include the directories first, so we headers can
# be found in the subdirectory projects as well
include_directories(hatchetfish/include)
include_directories(damselfish/include)
include_directories(fluxions/include)
include_directories(fluxions-gte/include)
include_directories(fluxions-base/include)
include_directories(starfish/include)
include_directories(include)

add_subdirectory(hatchetfish)
add_subdirectory(damselfish)
add_subdirectory(fluxions-gte)
add_subdirectory(fluxions-base)
add_subdirectory(fluxions)
add_subdirectory(starfish)

add_executable(${PROJECT_NAME}
	sunfish.cpp
	)

target_link_libraries(${PROJECT_NAME} PUBLIC fluxions)
target_link_libraries(${PROJECT_NAME} PUBLIC starfish)
target_link_libraries(${PROJECT_NAME} PUBLIC fluxions-base)
target_link_libraries(${PROJECT_NAME} PUBLIC fluxions-gte)
target_link_libraries(${PROJECT_NAME} PUBLIC damselfish)
target_link_libraries(${PROJECT_NAME} PUBLIC hatchetfish)

find_package(GLEW REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE GLEW::GLEW)

find_package(SDL2 CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE SDL2::SDL2main SDL2::SDL2-static)

find_package(sdl2-image CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE SDL2::SDL2_image)

find_package(sdl2-mixer CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE SDL2::SDL2_mixer)
