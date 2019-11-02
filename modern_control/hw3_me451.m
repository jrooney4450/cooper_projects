close all; clear; clc;

% choose one of the three paths below

path = pathLaneChange();
xmin = 0;
xmax = 100;
ymin = -15;
ymax = 15;
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
controller.DesiredLinearVelocity = 5;
controller.MaxAngularVelocity = 3;
controller.LookaheadDistance = 3;

goalRadius = 1;
distanceToGoal = norm(robotInitialLocation - robotGoal);

% Initialize the simulation loop
sampleTime = 0.1;
vizRate = rateControl(1/sampleTime);

% Initialize the figure
% figure(1)
figure(2)
title("Station vs. Cross Track Error")
xlabel("Station (m)")
ylabel("Cross Track Error (m)")

% Determine vehicle frame size to most closely represent vehicle with plotTransforms
frameSize = robot.TrackWidth/0.2;

% % Array form
% station = zeros(1,1);
% crossTrackError = zeros(1,1);

% Float form
station = 0;
crossTrackError = 0;

% Point counter initialization
counter = 1;

while( distanceToGoal > goalRadius )
    % Compute the controller outputs, i.e., the inputs to the robot
    [v, omega] = controller(robotCurrentPose);

    % Get the robot's velocity using controller inputs
    vel = derivative(robot, robotCurrentPose, [v omega]);
    
    prevPose = robotCurrentPose;
    
    % Update the current pose
    robotCurrentPose = robotCurrentPose + vel*sampleTime; 
    
    % Float form of station
    station = station + sqrt((robotCurrentPose(1) - prevPose(1))^2 + (robotCurrentPose(2) - prevPose(2))^2);
    
% %     Array form of station
%     stationAdd = [station(counter) + sqrt((robotCurrentPose(1) - prevPose(1))^2 + (robotCurrentPose(2) - prevPose(2))^2)];
%     station = [station stationAdd];
    
    % Calculate cross-track error
    pointBefore = path(counter,:);
    pointAfter = path(counter+1,:);
    pointTwo = path(counter+2,:);
    
    % Determine which point is closer
    % Distance from robot current state to previous point on path
    distanceBefore = sqrt((robotCurrentPose(1) - pointBefore(1))^2 + (robotCurrentPose(2) - pointBefore(2))^2);
    
    % Distance from robot current state to two points ahead
    distanceTwo = sqrt((robotCurrentPose(1) - pointTwo(1))^2 + (robotCurrentPose(2) - pointTwo(2))^2);
    
    % Once distance to two points ahead becomes larger than the distance to
    % the previous point, increment in the path points
    if distanceTwo <= distanceBefore 
        counter = counter + 1;
        pointBefore = path(counter,:);
        pointAfter = path(counter+1,:);
        pointTwo = path(counter+2,:);
        disp("the closest point changed");
    end
    
    % Find the distance from the robotState two the line formed between the
    % two most recent path points
    pathSlope = (pointBefore(2) - pointAfter(2)) / (pointBefore(1) - pointAfter(1));
    pathIntersect = pointBefore(2) - (pathSlope * pointBefore(1));
%     robotSlope = tan(robotCurrentPose(3));
%     robotIntersect = robotCurrentPose(2) - (robotSlope * robotCurrentPose(1));
%     robotSlopePerp = -1/robotSlope;
%     xIntersect = pathIntersect - robotIntersect / (robotSlopePerp - pathSlope);
%     yIntersect = robotSlope*xIntersect + robotSlope;

    % Find the equation of the line in Hesse normal form
    a = -pathSlope;
    b = 1;
    c = -pathIntersect;
%     den = (c/abs(c)) * sqrt(a^2 + b^2);
%     a = a/den;
%     b = b/den;
%     c = c/den;
    x0 = robotCurrentPose(1);
    y0 = robotCurrentPose(2);
    crossTrackError = (abs(a*x0 + b*y0 + c)) / sqrt(a^2 + b^2);
    
%     % Float form of cross-track error
%     crossTrackError = sqrt((robotCurrentPose(1) - xIntersect)^2 + (robotCurrentPose(2) - yIntersect)^2);
    
%     % Array form of cross-track error
%     crossTrackErrorAdd = [sqrt((robotCurrentPose(1) - xIntersect)^2 + (robotCurrentPose(2) - yIntersect)^2)];
%     crossTrackError = [crossTrackError crossTrackErrorAdd];
    
    % Re-compute the distance to the goal
    distanceToGoal = norm(robotCurrentPose(1:2) - robotGoal(:));
    
    % Update the plot
    hold off
    
    % Plot path each instance so that it stays persistent while robot mesh
    % moves
    figure(1)
    plot(path(:,1), path(:,2),"k--d")
    hold all
    
    % Plot the path of the robot as a set of transforms
    plotTrVec = [robotCurrentPose(1:2); 0];
    plotRot = axang2quat([0 0 1 robotCurrentPose(3)]);
    plotTransforms(plotTrVec', plotRot, "MeshFilePath", "groundvehicle.stl", "Parent", gca, "View","2D", "FrameSize", frameSize);
%     light;

    xlim([xmin xmax])
    ylim([ymin ymax])
    
%     % Plot heading versus cross track error
%     figure(2)
%     hold all
%     plot(station(counter), crossTrackError(counter), '-x');

    figure(2)
    hold all
    plot(station, crossTrackError, '.');
    
    waitfor(vizRate);
end

% figure(3)
% plot(station, crossTrackError);

function path = pathLaneChange()
    % Lane change path
    startLen = 30;
    endLen = 60;
    laneChangeAngle = deg2rad(10); % Angle at which to take the lane change
    laneWidth = 3;                 % Common roadway lane width in meters
    laneChangeDist = laneWidth/tan(laneChangeAngle);
    
    path = [];
    x = 0:1:startLen; % Increment by a meter
    for i = 1:length(x)
        addMe = [i, 0];
        path = cat(1, path, addMe); % add segment to total path 
    end
    
    x = 0:1:laneChangeDist;
    xunit = x*cos(laneChangeAngle);
    yunit = x*sin(laneChangeAngle);
    for i = 1:length(x)
        addMe = [startLen + xunit(i), yunit(i)];
        path = cat(1, path, addMe); % add segment to total path 
    end

    x = 0:1:endLen; % Increment by a meter
    for i = 1:length(x)
        addMe = [startLen + xunit(end) + i, yunit(end)];
        path = cat(1, path, addMe); % add segment to total path 
    end
end

function path = pathFigureEight()
    % Figure 8 path (sideways) with radius r and first circle at (x, y)
    path = [];
    r = 60;
    x = 60;
    y = 60;
    segmentLen = 50; % Set the number of waypoints along each semi-circle
    offset = 0.01; % Sets an offset between semi-circle transitions

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