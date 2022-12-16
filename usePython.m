function d_data = usePython(in1, in2, in3, in5, in4, in6, in7)
    coder.extrinsic('py.matpy.soc');
    data = 0;
    data = py.matpy.soc(in1, in2, in3, in5, in4, in6, in7);
    data = int64(data);
    d_data = double(data);
end