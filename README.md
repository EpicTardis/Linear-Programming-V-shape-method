# GPU Parallel Computing in Solving Linear Programming Problems
  We use V-shape method, a mathematical trick in removing redundancy lines to speed up computing process. Meanwhile, we minimize memory exchange between CPU and GPU since the process would take long time(compared with computing time).
  This program takes about 600ms to solve a problem consisting of 5 million lines on GTX960M.
