close all; clear; clc;

% choose one of the three paths below

path = pathLaneChange();
xmin = 0;
xmax = 500;
ymin = -30;
ymax = 30;
initialOrientation = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%

% path = pathFigureEight();
% xmin = -10;
% xmax = 250;
% ymin = -10;
% ymax = 130;
% initialOrientation = -pi/2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%

% path = pathRoad();
% xmin = -10;
% xmax = 250;
% ymin = -10;
% ymax = 130;
% initialOrientation = -pi/2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%

robotInitialLocation = path(1,:);
robotGoal = path(end,:);
robotCurrentPose = [robotInitialLocation initialOrientation]';

robot = differentialDriveKinematics("TrackWidth", 1, "VehicleInputs", "VehicleSpeedHeadingRate");

controller = controllerPurePursuit;
controller.Waypoints = path;
controller.DesiredLinearVelocity = 15;
controller.MaxAngularVelocity = 5;
controller.LookaheadDistance = 3;

goalRadius = 1;
distanceToGoal = norm(robotInitialLocation - robotGoal);

% Initialize the simulation loop
sampleTime = 0.1;
vizRate = rateControl(1/sampleTime);

% Initialize the figure
figure

% Determine vehicle frame size to most closely represent vehicle with plotTransforms
frameSize = robot.TrackWidth/0.2;

station = 0;

while( distanceToGoal > goalRadius )
    
    % Compute the controller outputs, i.e., the inputs to the robot
    [v, omega] = controller(robotCurrentPose);
    
    % Get the robot's velocity using controller inputs
    vel = derivative(robot, robotCurrentPose, [v omega]);
    
    station_prev = sqrt(robotCurrentPose(1)^2 + robotCurrentPose(2)^2);
    
    % Update the current pose
    robotCurrentPose = robotCurrentPose + vel*sampleTime; 
    
    station = station + (sqrt(robotCurrentPose(1)^2 + robotCurrentPose(2)^2) - station_prev);
    disp(station);
    
    % Find closest waypoint
    
    
    % Re-compute the distance to the goal
    distanceToGoal = norm(robotCurrentPose(1:2) - robotGoal(:));
    
    % Update the plot
    hold off
    
    % Plot path each instance so that it stays persistent while robot mesh
    % moves
    plot(path(:,1), path(:,2),"k--d")
    hold all
    
    % Plot the path of the robot as a set of transforms
    plotTrVec = [robotCurrentPose(1:2); 0];
    plotRot = axang2quat([0 0 1 robotCurrentPose(3)]);
    plotTransforms(plotTrVec', plotRot, "MeshFilePath", "groundvehicle.stl", "Parent", gca, "View","2D", "FrameSize", frameSize);
    light;

    xlim([xmin xmax])
    ylim([ymin ymax])
    
    waitfor(vizRate);
end

function path = pathLaneChange()
    % Lane change path
    laneChangeAngle = deg2rad(10); % Angle at which to take the lane change
    laneWidth = 3;                 % Common roadway lane width in meters
    laneChangeDist = laneWidth/tan(laneChangeAngle);
    
    path = [0.00                   0.00;       % Start point
            225.00                 0.00;       % Start of lane change
            225+laneChangeDist      laneWidth;  % End of lane change
            225+laneChangeDist+200   laneWidth]; % Continue in new lane
end

function path = pathFigureEight()
    % Figure 8 path (sideways) with radius r and first circle at (x, y)
    path = [];
    r = 60;
    x = 60;
    y = 60;
    segmentLen = 10; % Set the number of waypoints along each semi-circle
    offset = 0.1; % Sets an offset between semi-circle transitions

    % Plot and get path for first semi circle
    th = -pi:pi/segmentLen:0-offset;
    xunit = r * cos(th) + x;
    yunit = r * sin(th) + y;
    for i = 1:length(th)
        addMe = [xunit(i), yunit(i)];
        path = cat(1, path, addMe); % add segment to total path 
    end
    plot(xunit, yunit);

    % Plot and get path for second semi circle
    x = x + 2*r; % Move x to the origin of an adjacent semi-circle
    th = pi+offset:pi/segmentLen:2*pi-offset;
    xunit = r * cos(th) + x;
    yunit = r * -sin(th) + y;
    for i = 1:length(th)
        addMe = [xunit(i), yunit(i)];
        path = cat(1, path, addMe); % add segment to total path 
    end

    % Plot and get path for third semi circle
    th = 0+offset:pi/segmentLen:pi-offset;
    xunit = r * cos(th) + x;
    yunit = r * -sin(th) + y;
    for i = 1:length(th)
        addMe = [xunit(i), yunit(i)];
        path = cat(1, path, addMe); % add segment to total path 
    end

    % Plot and get path for fourth semi circle
    x = x - 2*r; % Move x back to the origin of the first semi-circle
    th = 0+offset:pi/segmentLen:pi; % 
    xunit = r * cos(th) + x;
    yunit = r * sin(th) + y;
    for i = 1:length(th)
        addMe = [xunit(i), yunit(i)];
        path = cat(1, path, addMe); % add segment to total path 
    end
end

function path = pathRoad()
    % Road path with random curves and straight points
    path = [];
    r = 60;
    x = 60;
    y = 60;
    segmentLen = 10; % Set the number of waypoints along each semi-circle
    offset = 0.1; % Sets an offset between semi-circle transitions
    
    % Plot and get path for a first semi circle
    th = -pi:pi/segmentLen:0-offset;
    xunit = r * cos(th) + x;
    yunit = r * sin(th) + y;
    for i = 1:length(th)
        addMe = [xunit(i), yunit(i)];
        path = cat(1, path, addMe); % add segment to total path 
    end
    plot(xunit, yunit);
    
    x = 140;
    addMe = [x, y];
    path = cat(1, path, addMe);
    
    x = 210;
    r = 40;
    % Plot and get path for a first semi circle
    th = -pi:pi/segmentLen:0-offset;
    xunit = r * cos(th) + x;
    yunit = r * sin(th) + y;
    for i = 1:length(th)
        addMe = [xunit(i), yunit(i)];
        path = cat(1, path, addMe); % add segment to total path 
    end
    plot(xunit, yunit);
    
    % Add a few more straight regions
    addMe = [200, 100];
    path = cat(1, path, addMe);
    addMe = [120, 120];
    path = cat(1, path, addMe);
    addMe = [50,  100];
    path = cat(1, path, addMe);
    addMe = [0,   0];
    path = cat(1, path, addMe);
end