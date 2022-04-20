from setuptools import setup

setup(name="gym_maze",
      version="0.4",
      url="https://github.com/tuzzer/gym-maze",
      author="Matthew T.K. Chan",
      license="MIT",
      packages=["gym_baby", "gym_baby.envs"],
      package_data = {
          "gym_baby.envs": ["maze_samples/*.npy"]
      },
      install_requires = ["gym", "pygame", "numpy"]
)
