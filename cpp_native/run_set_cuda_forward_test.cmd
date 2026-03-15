@echo off
setlocal

"C:\WS\Test\3DGS-to-PC\cpp_native\build_cuda_x64\gs2pc_cli.exe" ^
  --input_path "C:\WS\Test\3DGS-to-PC\set\iteration_7000\point_cloud.ply" ^
  --transform_path "C:\WS\Test\3DGS-to-PC\set\050702\sparse\0" ^
  --output_path "C:\WS\Test\3DGS-to-PC\set\iteration_7000\point_cloud_cuda_forward_test.ply" ^
  --num_points 1000000 ^
  --max_sh_degree 3 ^
  --colour_quality high

endlocal
