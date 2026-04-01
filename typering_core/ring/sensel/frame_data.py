import sys
import cv2
import time
import numpy as np


class ContactData():
    ''' Contains all info in one keystroke
    Attrs:
        id: a unique integer to number a keystroke with upper limit 16
        state: 1 => the start of a keystroke, 2 => middle, 3 => the end
            number of states that equal 1 or 3 can only be 1
        label: -1 => unsure, 0 => negative, 1 => positive
    '''
    

    def __init__(self, id=0, state=0, x=0, y=0, area=0, force=0, major=0, minor=0,
            delta_x=0, delta_y=0, delta_force=0, delta_area=0, label=-1):
        self.id = id
        self.state = state
        self.x = x
        self.y = y
        self.area = area
        self.force = force
        self.major = major
        self.minor = minor
        self.delta_x = delta_x
        self.delta_y = delta_y
        self.delta_force = delta_force
        self.delta_area = delta_area
        self.label = label  # unintentional touch label
        

    def __str__(self):
        contact_info = 'id: ' + str(self.id) + '\n' + 'state: ' + str(self.state) + '\n' \
            + 'x: ' + str(self.x) + '\n' + 'y: ' + str(self.y) + '\n' \
            +'area: ' + str(self.area) + '\n' + 'force: ' + str(self.force) + '\n' \
            + 'major: ' + str(self.major) + '\n' + 'minor: ' + str(self.minor) + '\n' \
            + 'delta_x: ' + str(self.delta_x) + '\n' + 'delta_y: ' + str(self.delta_y) + '\n' \
            + 'delta_force: ' + str(self.delta_force) + '\n' \
            + 'delta_area: ' + str(self.delta_area) + '\n' + 'label: ' + str(self.label)
        return contact_info
    

class FrameData():
    ''' info in one frame of morph board
    attrs:
        force_array: TODO
        contacts: list of keystroke contacts on the board
        timestamp: time acquired from time.perf_counter for each frame
    '''


    def __init__(self, force_map, timestamp):
        self.force_map = force_map
        self.timestamp = timestamp
        self.contacts = []

    
    def append_contact(self, contact):
        self.contacts.append(contact)
        
        
    def render(self) -> np.ndarray:
        ''' Render the frame data into an image.
        Returns:
            np.ndarray, the rendered image.
        '''
        upsample_ratio = 4
        H, W = self.force_map.shape
        force_map = cv2.resize(self.force_map, (W * upsample_ratio, H * upsample_ratio))
        force_map = cv2.cvtColor(force_map, cv2.COLOR_GRAY2BGR)
        force_map = np.clip(force_map * 255, 0, 255).astype(np.uint8)
        
        unsure_color = (255, 0, 0)
        negative_color = (0, 0, 255)
        positive_color = (0, 255, 0)
        for contact in self.contacts:
            if contact.label == 1:
                color = positive_color
            elif contact.label == 0:
                color = negative_color
            else: color = unsure_color
            cv2.circle(force_map, (int((contact.x * W + 0.5) * upsample_ratio),
                int((contact.y * H + 0.5) * upsample_ratio)), 8, color, -1)
        return force_map