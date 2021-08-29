import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import line
from PIL import Image
from line import Line

left_line = Line()
right_line = Line()

def do_canny(frame):
    # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    # Applies a 5x5 gaussian blur with deviation of 0 to frame - not mandatory since Canny will do this for us
    blur = cv.GaussianBlur(gray, (3, 3), 0)
    # Applies Canny edge detector with minVal of 50 and maxVal of 150
    canny = cv.Canny(blur, 30, 80)
    return canny

def do_segment(frame):
    # Since an image is a multi-directional array containing the relative intensities of each pixel in the image, we can use frame.shape to return a tuple: [number of rows, number of columns, number of channels] of the dimensions of the frame
    # frame.shape[0] give us the number of rows of pixels the frame has. Since height begins from 0 at the top, the y-coordinate of the bottom of the frame is its height
    height = frame.shape[0]
    widht = frame.shape[1]
    # Creates a triangular polygon for the mask defined by three (x, y) coordinates
    polygons = np.array([
                           [(0, 210), (0, 310), (260,200), (300,200),(widht, 300), (widht,210)]
                        ])
    # Creates an image filled with zero intensities with the same dimensions as the frame
    mask = np.zeros_like(frame)
    # Allows the mask to be filled with values of 1 and the other areas to be filled with values of 0
    cv.fillPoly(mask, polygons, 255)
    # A bitwise and operation between the mask and frame keeps only the triangular area of the frame
    segment = cv.bitwise_and(frame, mask)
    return segment

def calculate_lines(frame, lines):
    # Empty arrays to store the coordinates of the left and right lines
    left = []
    right = []
    #print("hough:", lines)
    # Loops through every detected line
    for line in lines:
        # Reshapes line from 2D array to 1D array
        x1, y1, x2, y2 = line.reshape(4)

        # Fits a linear polynomial to the x and y coordinates and returns a vector of coefficients which describe the slope and y-intercept
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        y_intercept = parameters[1]
        # If slope is negative, the line is to the left of the lane, and otherwise, the line is to the right of the lane
        if slope < 0:
            left.append((slope, y_intercept))
        else:
            right.append((slope, y_intercept))
    # Averages out all the values for left and right into a single slope and y-intercept value for each line
    left_avg = np.average(left, axis = 0)
    right_avg = np.average(right, axis = 0)
    # Calculates the x1, y1, x2, y2 coordinates for the left and right lines
    left_line = calculate_coordinates(frame, left_avg)
    right_line = calculate_coordinates(frame, right_avg)
    intersections = getIntersection(left_line, right_line)
    x_in = intersections[0]
    #y_in = intersections[1]
    # print("left:",left_line)
    # print(left_line[1])
    #x2 = frame.shape[1]/2 - 80
    x_coor = np.arange(0, x_in)
    x = np.array([left_line[0], left_line[2]])
    y = np.array([left_line[1], left_line[3]])
    y_coor = np.interp(x_coor, x, y)
    coordinates = np.column_stack((x_coor, y_coor))

    #print(coordinates[-1])
    left = np.array([coordinates[0],coordinates[-1]])
    left = np.reshape(left, (1, 4))
    #print(left[0],left_line)
    #x2 = frame.shape[1]/2 + 50
    x_coor = np.arange(x_in, frame.shape[1])
    #print(right_line)
    x = np.array([right_line[2], right_line[0]])
    y = np.array([right_line[3], right_line[1]])
    #print("x",x, "y:",y)
    y_coor = np.interp(x_coor, x, y)
    coordinates = np.column_stack((x_coor, y_coor))
    # print(coordinates[0])
    # print(coordinates[-1])
    right = np.array([coordinates[0],coordinates[-1]])
    right = np.reshape(right, (1, 4))
    #print(left[0], right[0])
    return np.array([left[0], right[0]])
    return np.array([left_line, right_line])

def calculate_coordinates(frame, parameters):
    slope, intercept = parameters
    #print(slope, intercept)
    # Sets initial y-coordinate as height from top down (bottom of the frame)
    y1 = frame.shape[0]
    # Sets final y-coordinate as 150 above the bottom of the frame
    y2 = int(y1 - 480)
    # Sets initial x-coordinate as (y1 - b) / m since y1 = mx1 + b
    x1 = int((y1 - intercept) / slope)
    # Sets final x-coordinate as (y2 - b) / m since y2 = mx2 + b
    x2 = int((y2 - intercept) / slope)
    # get coordinates

    return np.array([x1, y1, x2, y2])

def visualize_lines(frame, lines):
    # Creates an image filled with zero intensities with the same dimensions as the frame
    lines_visualize = np.zeros_like(frame)
    # print(lines)
    # print(lines[0][2])

    # Checks if any lines are detected
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            # Draws lines between two coordinates with green color and 5 thickness
            cv.line(lines_visualize, (int(np.float32(x1)), int(np.float32(y1))), (int(np.float32(x2)), int(np.float32(y2))), (0, 0, 255), thickness=4)
            #cv.line(lines_visualize, (int(np.float32(lines[0][2])), int(np.float32(lines[0][3]))), (int(np.float32(lines[1][0])), int(np.float32(lines[1][1]))), (45, 255, 255), thickness=10)
    return lines_visualize

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv.getPerspectiveTransform(rect, dst)
	warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped

def getIntersection(line1, line2):
    #print(line1, line2)
    s1 = np.array([line1[0],line1[1]])

    e1 = np.array([line1[2],line1[3]])

    s2 = np.array([line2[0],line2[1]])
    e2 = np.array([line2[2],line2[3]])

    a1 = (s1[1] - e1[1]) / (s1[0] - e1[0])
    b1 = s1[1] - (a1 * s1[0])

    a2 = (s2[1] - e2[1]) / (s2[0] - e2[0])
    b2 = s2[1] - (a2 * s2[0])

    x = (b2 - b1) / (a1 - a2)
    y = a1 * x + b1

    return (x, y)

def illustrate_driving_lane_with_topdownview(image, left_line, right_line):
    """
    #---------------------
    # This function illustrates top down view of the car on the road.
    #  
    """

    rows, cols = image.shape[:2]
    window_img = np.zeros_like(image)

    window_margin = 56
    left_plotx, right_plotx = left_line, right_line
    ploty = left_line
    lane_width = right_line[0] - left_line[0]
    lane_center = (right_line[0] + left_line[0]) / 2
    lane_offset = cols / 2 - (2*left_line[0] + lane_width) / 2
    car_offset = int(lane_center - 360)
    
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([right_plotx + lane_offset - lane_width - window_margin / 4, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_plotx + lane_offset - lane_width+ window_margin / 4, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_plotx + lane_offset - window_margin / 4, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_plotx + lane_offset + window_margin / 4, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv.fillPoly(window_img, np.int_([left_line_pts]), (140, 0, 170))
    cv.fillPoly(window_img, np.int_([right_line_pts]), (140, 0, 170))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([right_plotx + lane_offset - lane_width + window_margin / 4, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_plotx + lane_offset - window_margin / 4, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv.fillPoly(window_img, np.int_([pts]), (0, 160, 0))

    #window_img[10:133,300:360] = img
    road_map = Image.new('RGBA', image.shape[:2], (0, 0, 0, 0))
    window_img = Image.fromarray(window_img)
    road_map.paste(window_img, (0, 0))
    road_map = np.array(road_map)
    road_map = cv.resize(road_map, (95, 95))
    road_map = cv.cvtColor(road_map, cv.COLOR_BGRA2BGR)

    return road_map

# The video feed is read in as a VideoCapture object
cap = cv.VideoCapture("/Users/karel/Documents/Master/IPM-F1Tenth/assets/levine_straights.mp4")

img_array = []
i = 0
while (cap.isOpened()):
    # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
    ret, frame = cap.read()
    if ret == False:
        break
    width = 640
    height = 480
    dim = (width, height)
    
    # resize image
    resized = cv.resize(frame, dim, interpolation = cv.INTER_AREA)
    j = 0 
    
    cv.imwrite("frame"+ str(j) + ".jpg", resized)
    canny = do_canny(resized)
    #plt.imshow(canny)
    #plt.show()

    segment = do_segment(canny)
    # cv.imshow("canny", segment)
    # plt.imshow(segment)
    # plt.show()
    # plt.imshow(segment)
    # plt.show()
    try:
        hough = cv.HoughLinesP(segment, 2, np.pi / 180, 100, np.array([]), minLineLength = 120, maxLineGap = 100)
        cdst = cv.cvtColor(segment, cv.COLOR_GRAY2BGR)
        cdstP = np.copy(cdst)
        if hough is not None:
            for i in range(0, len(hough)):
                l = hough[i][0]
                cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
        
        #cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
        #print(hough)
        # Averages multiple detected lines from hough into one line for left border of lane and one line for right border of lane
        #print(hough)
        lines = calculate_lines(resized, hough)
        #print(lines)

        lines_visualize = visualize_lines(resized, lines)
        # Overlays lines on frame by taking their weighted sums and adding an arbitrary scalar value of 1 as the gamma argument
        # #print(lines[0][0],lines[0][1])
        x1 = int(lines[0][0])
        y1 = int(lines[0][1])
        A = (x1, y1)
        A_trans = (x1, y1 + 30)
        cv.circle(resized,(x1,y1),3,(0,255,0),5)
        font=cv.FONT_HERSHEY_SIMPLEX
        cv.putText(resized,'A',(x1,y1-10), font, 0.5,(0,255,0),1)
        x2 = int(lines[0][2])
        y2 = int(lines[0][3])
        cv.circle(resized,(x2,y2),3,(0,255,0),5)
        cv.putText(resized,'V',(x2,y2-10), font, 0.5,(0,255,0),1)
        # being start and end two points (x1,y1), (x2,y2)
        discrete_line_left = list(zip(*line(*(x1,y1), *(x2,y2))))
        D = discrete_line_left[len(discrete_line_left) - int(len(discrete_line_left)/5)]
        cv.circle(resized,D,3,(0,255,0),5)
        cv.putText(resized,'D',(D[0],D[1]-10), font, 0.5,(0,255,0),1)
        x = int(lines[1][2])
        y = int(lines[1][3])
        B = (x, y)
        B_trans = (x1, y1 + 30)
        cv.circle(resized,(x,y),3,(0,255,0),5)
        cv.putText(resized,'B',(x-10,y-10), font, 0.5,(0,255,0),1)
        discrete_line_right = list(zip(*line(*(x2,y2), *(x,y))))
        C = discrete_line_right[int(len(discrete_line_right)/5)]
        cv.circle(resized,C,3,(0,255,0),5)
        cv.putText(resized,'C',(C[0],C[1]-10), font, 0.5,(0,255,0),1)

        D_comma = np.array([discrete_line_left[len(discrete_line_left) - int(len(discrete_line_left)/5)][0], discrete_line_left[len(discrete_line_left) - int(len(discrete_line_left)/5)][1]])
        cv.circle(resized,D_comma,3,(255,0,0),5)
        cv.putText(resized,'D\'',(D_comma[0],D_comma[1]-10), font, 0.5,(255,0,0),1)

        C_comma = np.array([discrete_line_right[int(len(discrete_line_right)/5)][0], discrete_line_right[int(len(discrete_line_right)/5)][1]])
        cv.circle(resized,C_comma,3,(255,0,0),5)
        cv.putText(resized,'C\'',(C_comma[0],C_comma[1]-10), font, 0.5,(255,0,0),1)

        A_comma = discrete_line_left[int(len(discrete_line_left)/2)]
        cv.circle(resized,A_comma,3,(255,0,0),5)
        cv.putText(resized,'A\'',(A_comma[0],A_comma[1]-10), font, 0.5,(255,0,0),1)

        B_comma =  np.array([discrete_line_right[len(discrete_line_right) - int(len(discrete_line_right)/2)][0], discrete_line_right[len(discrete_line_right) - int(len(discrete_line_right)/2)][1]])
        cv.circle(resized,B_comma,3,(255,0,0),5)
        cv.putText(resized,'B\'',(B_comma[0],B_comma[1]-10), font, 0.5,(255,0,0),1)
        #cv.line(resized, (int(np.float32(x1)), int(np.float32(y1))), (int(np.float32(discrete_line_left[len(discrete_line_left) - int(len(discrete_line_left)/3)][0])), int(np.float32(discrete_line_left[len(discrete_line_left) - int(len(discrete_line_left)/3)][1]))), (45, 255, 255), thickness=10)
        
        pts = np.array([(0,height - 60), A_comma, B_comma, (width, height - 60)])
        pts_transform = np.array([A_trans ,(D_comma[0]+30, D_comma[1]),(C_comma[0]+30,C_comma[1]),B_trans])
        #pts = np.array([(0,height), A ,D_comma,C_comma,B,(width, height)])
        overlay = resized.copy()
        cv.fillPoly(overlay, np.int_([pts]), (255, 0, 0))
        alpha = 0.5  # Transparency factor.

        #cv.circle(resized,(x,y),5,(255,0,255),3)
        # Following line overlays transparent rectangle over the image
        image_new = cv.addWeighted(overlay, alpha, resized, 1 - alpha, 0)
        #cv.circle(image_new,(int(x),int(y)),5,(255,0,255),3)
        output = cv.addWeighted(image_new, 0.9, lines_visualize, 1, 1)
        


        # Opens a new window and displays the output frame
        #cv.imshow("output", output)
        #plt.imshow(output)
        # plt.show()

        birdeye_view_panel = np.zeros_like(output)
        result = output.copy()
        #info_panel[5:110, 5:325] = (255, 255, 255)
        warped = four_point_transform(resized, pts)
        birdeye_view_panel = warped.copy()
        #cv.imshow('road info', warped)

        scale_percent = 20
        width_bev = int(birdeye_view_panel.shape[1] * scale_percent / 100)
        height_bev = int(birdeye_view_panel.shape[0] * scale_percent / 100)
        dim_bev = (175, 175)
        birdeye_view_panel = cv.resize(birdeye_view_panel, dim_bev, interpolation = cv.INTER_AREA)
        print(birdeye_view_panel.shape)
        result[5:180, width-181:width-6,:] = birdeye_view_panel
        #road_map = illustrate_driving_lane_with_topdownview(output, lines[0], lines[1])
        #birdeye_view_panel[10:105, width-106:width-11] = road_map

        cv.imshow('road info', result)
        height, width, layers = result.shape
        size = (width,height)
        i = i + 1
        cv.imwrite('output_imgs/lines_alternative'+ str(i) + '.jpg', result) 
        img_array.append(result)
    except Exception as e: 
        print(e)

    # plt.imshow(output)
    # plt.show()
    # Make video
    # Frames are read by intervals of 10 milliseconds. The programs breaks out of the while loop when the user presses the 'q' key
    if cv.waitKey(10) & 0xFF == ord('q'):
        break

# The following frees up resources and closes all windows
cap.release()
cv.destroyAllWindows()

out = cv.VideoWriter('lines_edges_detector_alternative.mp4',cv.VideoWriter_fourcc(*'MP4V'), 30, size)
for i in range(len(img_array)):
    out.write(img_array[i])


