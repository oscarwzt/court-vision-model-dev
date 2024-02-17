COLORS = {
    "hoop_box": (0, 255, 0),          # Green
    "detection_area": (255, 0, 0), # Blue
    "entry_box": (0, 0, 255),      # Red
    "exit_box": (0, 255, 255),     # Yellow
    "basketball": (255, 255, 0),   # Cyan
    "person": (255, 0, 255)        # Magenta
}

def get_detection_box(x1, y1, x2, y2, n_height = 3.5, n_width = 2):
    # Calculate the center of the box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # Calculate new width and height
    new_width = (x2 - x1) * n_width
    new_height = (y2 - y1) * n_height

    # Calculate new coordinates and round them to the nearest integer
    new_x1 = int(round(center_x - new_width / 2))
    new_y1 = int(round(center_y - new_height / 2))
    new_x2 = int(round(center_x + new_width / 2))
    new_y2 = int(round(center_y + new_height / 2))

    return new_x1, new_y1, new_x2, new_y2

def get_entry_box(x1, y1, x2, y2, alpha = 0.2, beta = -0.2):
    box_height = y2 - y1
    box_width = x2 - x1
    
    y1_, y2_ = y1 - int(box_height * (1 + beta)), y1
    x1_, x2_ = x1 - int(box_width * alpha), x2 + int(box_width * alpha)
    
    return x1_, y1_, x2_, y2_

def get_exit_box(x1, y1, x2, y2, alpha = 0.2, beta = -0.2):
    box_height = y2 - y1
    box_width = x2 - x1
    
    y1_, y2_ = y2, y2 + int(box_height * (1 + beta))
    x1_, x2_ = x1 - int(box_width * alpha), x2 + int(box_width * alpha)
    
    return x1_, y1_, x2_, y2_

def is_in_box(x_center, y_center, x1, y1, x2, y2):
    if x1 <= x_center <= x2 and y1 <= y_center <= y2:
        return True
    return False

def get_center(x1, y1, x2, y2):
    return (x1 + x2) // 2, (y1 + y2) // 2