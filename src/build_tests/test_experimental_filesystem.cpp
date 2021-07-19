#include <experimental/filesystem>
#include <iostream>

namespace fs = std::experimental::filesystem;

int main(int argc, char* argv[]) {
  fs::path mypath = {"./"};

  std::cout << "Does directory 'test_dir' already exist? ";
  if (fs::is_directory("test_dir")) {
    std::cout << "Yes" << std::endl;
    std::cout << "Not touching it, exiting..." << std::endl;
    return 0;
  } else {
    std::cout << "No" << std::endl;
  }

  std::cout << "Creating directory 'test_dir'...";
  fs::create_directory("test_dir");
  std::cout << "done!" << std::endl;

  std::cout << "Does directory 'test_dir' exist? ";
  if (fs::is_directory("test_dir")) {
    std::cout << "Yes" << std::endl;
  } else {
    std::cout << "No" << std::endl;
  }

  std::cout << "Removing directory 'test_dir'...";
  fs::remove("test_dir");
  std::cout << "done!" << std::endl;
  return 0;
}