% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Copyright (C) OMG Plc 2009.
% All rights reserved.  This software is protected by copyright
% law and international treaties.  No part of this software / document
% may be reproduced or distributed in any form or by any means,
% whether transiently or incidentally to some other use of this software,
% without the written permission of the copyright owner.
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Part of the Vicon DataStream SDK for MATLAB.
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Program options
TransmitMulticast = false;
EnableHapticFeedbackTest = false;
HapticOnList = {'ViconAP_001';'ViconAP_002'};
bReadCentroids = false;
bReadRays = false;
axisMapping = 'ZUp';

% A dialog to stop the loop
MessageBox = msgbox( 'Stop DataStream Client', 'Vicon DataStream SDK' );

% Load the SDK
fprintf( 'Loading SDK...' );
Client.LoadViconDataStreamSDK();
fprintf( 'done\n' );

% Program options
HostName = 'localhost:801';

% Make a new client
MyClient = Client();

% Connect to a server
fprintf( 'Connecting to %s ...', HostName );
while ~MyClient.IsConnected().Connected
  % Direct connection
  MyClient.Connect( HostName );
  
  % Multicast connection
  % MyClient.ConnectToMulticast( HostName, '224.0.0.0' );
  
  fprintf( '.' );
end
fprintf( '\n' );

% Enable some different data types
MyClient.EnableSegmentData();
MyClient.EnableMarkerData();
MyClient.EnableUnlabeledMarkerData();
MyClient.EnableDeviceData();
if bReadCentroids
  MyClient.EnableCentroidData();
end
if bReadRays
  MyClient.EnableMarkerRayData();
end

fprintf( 'Segment Data Enabled: %s\n',          AdaptBool( MyClient.IsSegmentDataEnabled().Enabled ) );
fprintf( 'Marker Data Enabled: %s\n',           AdaptBool( MyClient.IsMarkerDataEnabled().Enabled ) );
fprintf( 'Unlabeled Marker Data Enabled: %s\n', AdaptBool( MyClient.IsUnlabeledMarkerDataEnabled().Enabled ) );
fprintf( 'Device Data Enabled: %s\n',           AdaptBool( MyClient.IsDeviceDataEnabled().Enabled ) );
fprintf( 'Centroid Data Enabled: %s\n',         AdaptBool( MyClient.IsCentroidDataEnabled().Enabled ) );
fprintf( 'Marker Ray Data Enabled: %s\n',       AdaptBool( MyClient.IsMarkerRayDataEnabled().Enabled ) );

% Set the streaming mode
MyClient.SetStreamMode( StreamMode.ClientPull );

% Set the global up axis
if axisMapping == 'XUp'
  MyClient.SetAxisMapping( Direction.Up, ...
                          Direction.Forward,      ...
                          Direction.Left ); % X-up
elseif axisMapping == 'YUp'
  MyClient.SetAxisMapping( Direction.Forward, ...
                         Direction.Up,    ...
                         Direction.Right );    % Y-up
else
  MyClient.SetAxisMapping( Direction.Forward, ...
                         Direction.Left,    ...
                         Direction.Up );    % Z-up
end

Output_GetAxisMapping = MyClient.GetAxisMapping();
fprintf( 'Axis Mapping: X-%s Y-%s Z-%s\n', Output_GetAxisMapping.XAxis.ToString(), ...
                                           Output_GetAxisMapping.YAxis.ToString(), ...
                                           Output_GetAxisMapping.ZAxis.ToString() );
  
if TransmitMulticast
  MyClient.StartTransmittingMulticast( 'localhost', '224.0.0.0' );
end  

% Initialize global translation matrix for plotting
global_trans_matrix = [];

Counter = 1;
% Loop until the message box is dismissed
while ishandle( MessageBox )
  drawnow;
  Counter = Counter + 1;
  
  % Get a frame
  fprintf( 'Waiting for new frame...' );
  while MyClient.GetFrame().Result.Value ~= Result.Success
    fprintf( '.' );
  end% while
  fprintf( '\n' );  

  % Get the frame number
  Output_GetFrameNumber = MyClient.GetFrameNumber();
  fprintf( 'Frame Number: %d\n', Output_GetFrameNumber.FrameNumber );

  % Get the frame rate
  Output_GetFrameRate = MyClient.GetFrameRate();
  fprintf( 'Frame rate: %g\n', Output_GetFrameRate.FrameRateHz );

  for FrameRateIndex = 1:MyClient.GetFrameRateCount().Count
    FrameRateName  = MyClient.GetFrameRateName( FrameRateIndex ).Name;
    FrameRateValue = MyClient.GetFrameRateValue( FrameRateName ).Value;

    fprintf( '%s: %gHz\n', FrameRateName, FrameRateValue );
  end% for  

  fprintf( '\n' );
  % Get the timecode
  Output_GetTimecode = MyClient.GetTimecode();
  fprintf( 'Timecode: %dh %dm %ds %df %dsf %s %d %d %d\n\n',    ...
                     Output_GetTimecode.Hours,                  ...
                     Output_GetTimecode.Minutes,                ...
                     Output_GetTimecode.Seconds,                ...
                     Output_GetTimecode.Frames,                 ...
                     Output_GetTimecode.SubFrame,               ...
                     AdaptBool( Output_GetTimecode.FieldFlag ), ...
                     Output_GetTimecode.Standard.Value,         ...
                     Output_GetTimecode.SubFramesPerFrame,      ...
                     Output_GetTimecode.UserBits );

  % Get the latency
  fprintf( 'Latency: %gs\n', MyClient.GetLatencyTotal().Total );
  
  for LatencySampleIndex = 1:MyClient.GetLatencySampleCount().Count
    SampleName  = MyClient.GetLatencySampleName( LatencySampleIndex ).Name;
    SampleValue = MyClient.GetLatencySampleValue( SampleName ).Value;

    fprintf( '  %s %gs\n', SampleName, SampleValue );
  end% for  
  fprintf( '\n' );
                     
  % Count the number of subjects
  SubjectCount = MyClient.GetSubjectCount().SubjectCount;
  
  for SubjectIndex = 1:SubjectCount
    fprintf( '  Subject #%d\n', SubjectIndex - 1 );
    
    % Get the subject name
    SubjectName = MyClient.GetSubjectName( SubjectIndex ).SubjectName;
    fprintf( '    Name: %s\n', SubjectName );
    
    % Get the root segment
    RootSegment = MyClient.GetSubjectRootSegmentName( SubjectName ).SegmentName;
    fprintf( '    Root Segment: %s\n', RootSegment );

    % Count the number of segments
    SegmentCount = MyClient.GetSegmentCount( SubjectName ).SegmentCount;
    fprintf( '    Segments (%d):\n', SegmentCount );
    for SegmentIndex = 1:SegmentCount
            
            % Get the segment index
            fprintf( '      Segment #%d\n', SegmentIndex - 1 );
    
            % Get the segment name
            SegmentName = MyClient.GetSegmentName( SubjectName, SegmentIndex ).SegmentName;
            fprintf( '        Name: %s\n', SegmentName );

            % Get the rotation and translation matrices from world frame 
            Output_GetSegmentGlobalTranslation = MyClient.GetSegmentGlobalTranslation( SubjectName, SegmentName );
            Output_GetSegmentGlobalRotationMatrix = MyClient.GetSegmentGlobalRotationMatrix( SubjectName, SegmentName );

            % Input into homogeneous transform
            homo = [Output_GetSegmentGlobalRotationMatrix.Rotation( 1 ), Output_GetSegmentGlobalRotationMatrix.Rotation( 2 ), Output_GetSegmentGlobalRotationMatrix.Rotation( 3 ), Output_GetSegmentGlobalTranslation.Translation( 1 ); 
                 Output_GetSegmentGlobalRotationMatrix.Rotation( 4 ), Output_GetSegmentGlobalRotationMatrix.Rotation( 5 ), Output_GetSegmentGlobalRotationMatrix.Rotation( 6 ), Output_GetSegmentGlobalTranslation.Translation( 2 );
                 Output_GetSegmentGlobalRotationMatrix.Rotation( 7 ), Output_GetSegmentGlobalRotationMatrix.Rotation( 8 ), Output_GetSegmentGlobalRotationMatrix.Rotation( 9 ), Output_GetSegmentGlobalTranslation.Translation( 3 );
                 0, 0, 0, 1];
                   
            % Get the yaw component of the euler angles
            Output_GetSegmentGlobalRotationEulerXYZ = MyClient.GetSegmentGlobalRotationEulerXYZ( SubjectName, SegmentName );
            euler = Output_GetSegmentGlobalRotationEulerXYZ.Rotation( 3 );
            
            % Create variable for each tracked object
            if convertCharsToStrings(SegmentName) == 'keyboard'
                homo_k = homo;
                euler_k = euler;
            elseif convertCharsToStrings(SegmentName) == 'robot'
                homo_r = homo;
                euler_r = euler;
            elseif convertCharsToStrings(SegmentName) == 'pencil'
                homo_s = homo;
                euler_s = euler;
            end

    end% SegmentIndex
  end% SubjectIndex
end% while true  

% Get the yaw angle offset between the keyboard and robot base
yaw_correct = euler_r - euler_k

% Get the keyboard in robot frame and extract translation offset
rHk = inv(homo_r)* homo_k;
x_off = (rHk(1,4))/1000
y_off = (rHk(2,4))/1000
z_off = (rHk(3,4))/1000;

% % Get the pencil in robot frame and extract translation offset
rHs = inv(homo_r) * homo_s;
x_rHs = (rHs(1,4))/1000;
y_rHs = (rHs(2,4))/1000;
z_rHs = (rHs(3,4))/1000;

% Get the difference in translation offset from the pencil to the C-key
x_del = x_off - x_rHs
y_del = y_off - y_rHs
z_del = z_off - z_rHs;

if TransmitMulticast
  MyClient.StopTransmittingMulticast();
end  

% Disconnect and dispose
MyClient.Disconnect();

% Unload the SDK
fprintf( 'Unloading SDK...' );
Client.UnloadViconDataStreamSDK();
fprintf( 'done\n' );

