## 1. reference page
1. [reference1](https://docs.prophesee.ai/stable/installation/windows_openeb.html)

## 2. clone specific code version
```bash
git clone https://github.com/prophesee-ai/openeb.git --branch 5.1.1
```

## 3.clone submodule
```bash
git submodule update --init --recursive
```

## 4. Cmake version
```bash
cmake --version
cmake version 3.26.6
```

## 5. Issues

1. The issue appeared in the openeb codebase where the vcpkg.json configuration file specified a specific version of vcpkg code when compiling boost-related libraries
2. Solution: use the latest version of vcpkg code and directly delete this section from vcpkg.json
```json
    "vcpkg-configuration" : {

        "default-registry": {

            "kind": "git",

            "repository": "https://github.com/Microsoft/vcpkg",

            "baseline": "a42af01b72c28a8e1d7b48107b33e4f286a55ef6"

        },

        "registries": [

            {

                 "kind": "git",

                 "repository": "https://github.com/Microsoft/vcpkg",

                 "baseline": "76d4836f3b1e027758044fdbdde91256b0f0955d",

                 "packages": [ "boost*", "boost-*"]

            }

        ]

    }
```

## 6. Install Python

```bash
python -m venv C:\Users\yao\Documents\courses\project\project_python\py3venv --system-site-packages
set PYTHONNOUSERSITE=true
C:\Users\yao\Documents\courses\project\project_python\py3venv\Scripts\python -m pip install pip --upgrade
C:\Users\yao\Documents\courses\project\project_python\py3venv\Scripts\python -m pip install -r C:\Users\yao\Documents\courses\project\openeb\utils\python\requirements_openeb.txt
```
## 7. Build/Compilation
```bash
mkdir build 
cd build
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=<OPENEB_SRC_DIR>\cmake\toolchains\vcpkg.cmake -DVCPKG_DIRECTORY=<VCPKG_SRC_DIR> -DBUILD_TESTING=OFF

cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release `-DCMAKE_TOOLCHAIN_FILE=C:\Users\yao\Documents\courses\project\openeb\cmake\toolchains\vcpkg.cmake ` -DVCPKG_DIRECTORY=C:\Users\yao\Documents\courses\project\vcpkg ` -DBUILD_TESTING=OFF
```
1. HDF5 Error
```bash
CMake Error at C:/Users/yao/Documents/courses/project/vcpkg/installed/x64-windows/share/hdf5/hdf5-targets.cmake:42 (message):
  Some (but not all) targets in this export set were already defined.

  Targets Defined: hdf5::h5diff

  Targets not yet defined: hdf5::hdf5-shared, hdf5::hdf5_tools-shared,
  hdf5::h5ls, hdf5::h5debug, hdf5::h5repart, hdf5::h5mkgrp, hdf5::h5clear,
  hdf5::h5delete, hdf5::h5import, hdf5::h5repack, hdf5::h5jam, hdf5::h5unjam,
  hdf5::h5copy, hdf5::h5stat, hdf5::h5dump, hdf5::h5format_convert,
  hdf5::h5perf_serial, hdf5::hdf5_hl-shared, hdf5::gif2h5, hdf5::h52gif,
  hdf5::h5watch, hdf5::hdf5_cpp-shared, hdf5::hdf5_hl_cpp-shared

Call Stack (most recent call first):
  C:/Users/yao/Documents/courses/project/vcpkg/installed/x64-windows/share/hdf5/hdf5-config.cmake:180 (include)
  C:/Program Files/CMake/share/cmake-3.26/Modules/FindHDF5.cmake:486 (find_package)
  sdk/modules/stream/cpp/3rdparty/hdf5_ecf/CMakeLists.txt:42 (find_package)


-- Configuring incomplete, errors occurred!
```
2. Solution
	1. Modify C:\Users\yao\Documents\courses\project\openeb\sdk\modules\stream\cpp\3rdparty\hdf5_ecf/CMakeLists.txt and replace line 42 with:
		```cmake
		if(NOT HDF5_FOUND)
		    find_package(HDF5 REQUIRED)
		endif()
		```
## 8. Continue CMake
```bash
cmake --build . --config Release --parallel 4
```

## 9. Environment Variables

```bash
C:\Users\yao\Documents\courses\project\openeb\build\utils\scripts>.\setup_env.bat
```

## 10. Install Driver Plugin
```bash
./wdi-simple.exe -n "EVK" -m "Prophesee" -v 0x04b4 -p 0x00f4
./wdi-simple.exe -n "EVK" -m "Prophesee" -v 0x04b4 -p 0x00f5
./wdi-simple.exe -n "EVK" -m "Prophesee" -v 0x04b4 -p 0x00f3
```

## 11. Test

```bash
cmake .. -A x64 -DCMAKE_TOOLCHAIN_FILE=C:\Users\yao\Documents\courses\project\openeb\cmake\toolchains\vcpkg.cmake -DVCPKG_DIRECTORY=C:\Users\yao\Documents\courses\project\vcpkg -DBUILD_TESTING=ON
```

## 12. Conclusion
1. Currently, all three cameras are connecting normally via the GUI interface
# 11-27 17:05 Installation - Modified openeb vcpkg.json
1. Used the vcpkg code package provided by the official website
2. Directly deleted this section from vcpkg.json
```json
    "vcpkg-configuration" : {

        "default-registry": {

            "kind": "git",

            "repository": "https://github.com/Microsoft/vcpkg",

            "baseline": "a42af01b72c28a8e1d7b48107b33e4f286a55ef6"

        },

        "registries": [

            {

                 "kind": "git",

                 "repository": "https://github.com/Microsoft/vcpkg",

                 "baseline": "76d4836f3b1e027758044fdbdde91256b0f0955d",

                 "packages": [ "boost*", "boost-*"]

            }

        ]

    }
```
# 11-26 16:44 Installation - Using MSVC 143, Windows 10/11 SDK

## 1. Unescaped Issue
1. Modified vcpkg.json file, added content
```json
{
  "dependencies": [
    "boost-thread"
  ]
}
```
3. user-config.jam handling - Not working, each installation will automatically generate a new configuration file which overwrites the changes
	1. Delete C:\Users\yao\Documents\courses\project\vcpkg-2024.04.26\buildtrees\boost-thread\x64-windows-dbg/user-config.jam
	2. Add quotes: <linkflags>"/machine:x64"
4. Copy a configuration file to the user directory:
	1. **Correct solution (recommended by official)**
	2. **Solution A: Place user-config.jam in user HOME directory (most stable, won't be overwritten)**
	3. Configuration file only needs one line of code:
	4. using msvc : 14.3 : "C:\\Program Files\\Microsoft Visual Studio\\2022\\BuildTools\\VC\\Tools\\MSVC\\14.44.35221\\bin\\Hostx64\\x64\\cl.exe" ;
5. Then re-execute vcpkg install --triplet x64-windows --x-install-root installed
	1. set VCVARSALL=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat  
	2. call "%VCVARSALL%"
	3. vcpkg install --triplet x64-windows --x-install-root installed

set VCVARSALL=C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat
call "%VCVARSALL%"
vcpkg install boost-thread:x64-windows


# 11-26 15:09 Installation - Using Latest Code Version

1. Same issue: the openeb vcpkg.json specifies the build code version for all boost* packages. It appears that this version uses msvc 140, but I have msvc 143 installed locally
## 1. Modified C:\Users\yao\Documents\courses\project\vcpkg-2024.04.26\buildtrees\boost-thread\x64-windows-dbg/user-config.jam

# 11-26 15:26 Installation - Using MSVC 140 Windows 10 SDK (Uninstalled Windows 11 SDK)

1. Fully followed the official documentation for installation

# 11-26 14:32 Installation - Using Official Website Package
## 1. Execute
```bash
.\vcpkg install --triplet x64-windows --x-install-root installed
```
## 2. Issues
```bash
notice: found boost-build.jam at C:/Users/yao/Documents/courses/project/vcpkg-2024.04.26/installed/x64-windows/tools/boost-build/boost-build.jam
notice: loading B2 from C:/Users/yao/Documents/courses/project/vcpkg-2024.04.26/installed/x64-windows/tools/boost-build/src/kernel/bootstrap.jam
notice: Site configuration files will be ignored due to the
notice: --ignore-site-config command-line option.
notice: Loading explicitly specified user configuration file:
    C:\Users\yao\Documents\courses\project\vcpkg-2024.04.26\buildtrees\boost-thread\x64-windows-dbg\user-config.jam
notice: Searching 'C:\Users\yao\Documents\courses\project\vcpkg-2024.04.26\buildtrees\boost-thread\x64-windows-dbg' for user-config configuration file 'user-config.jam'.
notice: Loading user-config configuration file 'user-config.jam' from 'C:/Users/yao/Documents/courses/project/vcpkg-2024.04.26/buildtrees/boost-thread/x64-windows-dbg'.
C:\Users\yao\Documents\courses\project\vcpkg-2024.04.26\buildtrees\boost-thread\x64-windows-dbg\user-config.jam:8: Unescaped special character in argument <linkflags>/machine:x64
```

## 3. Solutions
### 1. 👉 **Remove `registries`, use vcpkg default Boost (more stable), then reinstall Boost**

1. Modified vcpkg.json and removed registries:
```json
"vcpkg-configuration" : {
    "default-registry": {
        "kind": "git",
        "repository": "https://github.com/Microsoft/vcpkg",
        "baseline": "a42af01b72c28a8e1d7b48107b33e4f286a55ef6"
    }
}
```

2. vcpkg remove boost --recurse
3. vcpkg install --triplet x64-windows --x-install-root installed
4. Prompts:
	1. error: Unable to find file or target named
	2. error:     '/boost//C:/Users/yao/Documents/courses/project/vcpkg-2024.04.26/installed/x64-windows/debug/lib/boost_chrono-vc140-mt-gd.lib'
5. Use install command
```bash
vcpkg install boost-thread:x64-windows --triplet x64-windows
```
# 11-25 18:41 Installation

## 1. Execute
```bash
vcpkg install --triplet x64-windows
```
## 2. Issues
### 1. Config
`user-config.jam` file contains unescaped characters
Boost.Build is very sensitive to the colon `:` in `linkflags` for MSVC. If written directly as:
`<linkflags>/machine:x64`
it will report **Unescaped special character**. The correct way is:
`<linkflags>"\/machine:x64"`
or wrap it with **double quotes** to prevent b2 from parsing incorrectly.
### 2. Architecture/Address Model Confusion
The log shows:
`<address-model>64 <architecture>x86`
This means Boost is trying to compile 64-bit address model under **x86 architecture**, which is conflicting.  
Correct should be:
- `architecture=x86` + `address-model=32`  
    or
- `architecture=x86_64`/`amd64` + `address-model=64`
vcpkg's default `x64-windows` triplet is 64-bit, which indicates that your `user-config.jam` or b2 command line has an error where `architecture` is set to x86.
### 3. Cannot find library file path
The log shows:
`error: Unable to find file or target named '/boost//C:/Users/yao/.../boost_chrono-vc140-mt-gd.lib'`
This is because **Boost's stage/install path concatenation is incorrect**, causing b2 to try to parse the full path as a Boost library name.  
Common causes:
- `user-config.jam` has set incorrect `<library>` or `<include>` paths
- The triplet's `vcpkg.cmake` file specifies incorrect `BOOST_ROOT` or `BOOST_BUILD_PATH`
# Installation History
## 5. vcpkg Issues
### 1. The installation tutorial provided a package. I can extract and use the extracted package
1. Issue: executing the update command will prompt that there are no updates, and suggest using git pull to update, but since it's a package there is no .git, so there's nothing to update
```bash
vcpkg update
```
1. Currently continuing with this solution
### 2. I can also directly git clone the project code and then update it, but I'm not sure if I need to use the version of the code from the tutorial package

## 6. Execute Install Command

### 1. Using VS Developer Command Prompt
```bash
 vcpkg install --triplet x64-windows --x-install-root installed
```
1. Error, probably needs to use v140, but I have v143 installed locally, and also showing path issues?
### 2. Modified configuration file to resolve path issues
```bash
C:\Users\yao\Documents\courses\project\vcpkg-2024.04.26\buildtrees\boost-thread\x64-windows-dbg\user-config.jam

3. Modified the `<linkflags>` section, added quotes:
`<linkflags>" /machine:x64 " ;`
```
#### 1. Installation
```bash
vcpkg install boost-thread:x64-windows

error: In manifest mode, `vcpkg install` does not support individual package arguments.
```

#### 2. Use another installation method
1. Modified vcpkg.json file
```json
{
    "dependencies": [
        "libusb",
        "boost",
        "opencv",
        "dirent",
        "gtest",
        "pybind11",
        "glew",
        "glfw3",
        "boost-thread",
        { "name" : "hdf5" , "features" : ["cpp","threadsafe","tools","zlib"] },
        "protobuf"
    ],
    "vcpkg-configuration" : {
        "default-registry": {
            "kind": "git",
            "repository": "https://github.com/Microsoft/vcpkg",
            "baseline": "a42af01b72c28a8e1d7b48107b33e4f286a55ef6"
        },
        "registries": [
            {
                 "kind": "git",
                 "repository": "https://github.com/Microsoft/vcpkg",
                 "baseline": "76d4836f3b1e027758044fdbdde91256b0f0955d",
                 "packages": [ "boost*", "boost-*"]
            }
        ]
    }
}

```

### 3. Re-install to resolve version issue: should use v143 instead of v140
