
[own_path, ~, ~] = fileparts(mfilename('fullpath'));
module_path= py.sys.path;
if count(py.sys.path, module_path) == 0
    insert(py.sys.path,int32(0), module_path);
end

py.importlib.import_module('matlab_test')
py.matlab_test.myfunc1(py.int(0));
