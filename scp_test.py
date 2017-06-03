import paramiko


hostname = '150.214.59.226'
password = 'DircOwz1'
username = "jcmartinez"
port = 22

mypath='/media/jkmrto/Multimedia/VAN-applied-to-Nifti-images/test.py'
remotepath='/mnt/datos/home/jcmartinez/VAN-applied-to-Nifti-images/out/test.py'

remote_path_sweep_parameter_training_error  = """/mnt/datos/home/jcmartinez/VAN-applied-to-Nifti-images/out/
Sesion Sweep over svm minimum training error/loop_over_svm*"""
local_path_sweep_parameter_training_error = """ /media/jkmrto/Multimedia/VAN-applied-to-Nifti-images/sweep_parameters*"""

t = paramiko.Transport((hostname, 22))
t.connect(username=username, password=password)
sftp = paramiko.SFTPClient.from_transport(t)
sftp.get(remote_path_sweep_parameter_training_error,
         local_path_sweep_parameter_training_error)


