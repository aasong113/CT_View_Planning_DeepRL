from setuptools import setup

setup(name="gym_maze",
      version="0.4",
      url="https://github.com/tuzzer/gym-maze",
      author="Matthew T.K. Chan",
      license="MIT",
      packages=["gym_teen", "gym_teen.envs"],
      package_data = {
          "gym_teen.envs": ["maze_samples/*.npy"]
      },
      install_requires = ["gym", "pygame", "numpy"]
)
