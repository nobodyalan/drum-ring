import time
import pickle
import _thread
import numpy as np
from typing import List

from . import sensel
from .frame_data import FrameData, ContactData


class TypeBoard():
    ''' Manage the Sensel Morph board.
    '''


    def __init__(self, label_contact:bool=False, model_path:str='data/model/typeboard.model'):
        ''' Init TypeBoard.
        Args:
            label_contact (bool): whether to label each contact as unintentional or not.
            model_path (str): path to the TypeBoard model.
        '''
        self.MAX_X = 230.0
        self.MAX_Y = 130.0
        self.FPS = 50
        self.label_contact = label_contact  # whether to label contact or not
        if label_contact:
            self.scalar, self.clf = pickle.load(open(model_path, 'rb'))
        else: self.scalar, self.clf = None, None
        self._open_sensel()
        self._init_frame()
        self._init_history()


    def _open_sensel(self):
        handle = None
        (error, device_list) = sensel.getDeviceList()
        if device_list.num_devices != 0:
            (error, handle) = sensel.openDeviceByID(device_list.devices[0].idx)
        self.handle = handle
        sensel.setMaxFrameRate(self.handle, self.FPS)
        sensel.setScanDetail(self.handle, 1)


    def _init_frame(self):
        (error, self.info) = sensel.getSensorInfo(self.handle)
        error = sensel.setFrameContent(self.handle, 0x0F)
        error = sensel.setContactsMask(self.handle, 0x0F)
        error, frame = sensel.allocateFrameData(self.handle)
        error = sensel.startScanning(self.handle)
        self._frame = frame
        self.frames = []
        self.updated = False
        self.frame_cnt = 0
        try:
            _thread.start_new_thread(self._run, ())
        except:
            print("Thread Error")
            
    
    def _init_history(self):
        self.N = 16
        self.start_frame = np.zeros(self.N, dtype=np.int32)
        self.total_forces = []
        self.total_areas = []
        self.total_intens = []
        self.updated_frame_cnt = 0


    def _close_sensel(self):
        self.is_running = False
        error = sensel.freeFrameData(self.handle, self._frame)
        error = sensel.stopScanning(self.handle)
        error = sensel.close(self.handle)
    
    
    def _sync(self):
        if len(self.frames) > 0:
            while ((time.perf_counter() - self.frames[-1].timestamp) * self.FPS < 1):
                pass


    def _run(self):
        self.is_running = True
        timestamp = time.perf_counter()
        while self.is_running:
            error = sensel.readSensor(self.handle)
            (error, num_frames) = sensel.getNumAvailableFrames(self.handle)
            for i in range(num_frames):
                timestamp = time.perf_counter()
                error = sensel.getFrame(self.handle, self._frame)
            R = self.info.num_rows
            C = self.info.num_cols
            force_map = np.zeros((R, C), dtype=np.float32)
            for r in range(R):
                force_map[r, :] = self._frame.force_array[r * C : (r + 1) * C]
            force_map *= 0.05
            frame = FrameData(force_map, timestamp)

            for i in range(self._frame.n_contacts):
                c = self._frame.contacts[i]
                x = c.x_pos / self.MAX_X
                y = c.y_pos / self.MAX_Y
                contact = ContactData(c.id, c.state, x, y, c.area, c.total_force,
                    c.major_axis, c.minor_axis, c.delta_x, c.delta_y, c.delta_force, c.delta_area)
                frame.append_contact(contact)

            self.frames.append(frame)
            self.updated = True
        
        self._close_sensel()
        
    
    def _get_sequence(self, X):
        # a Faster Implementation of: return [np.mean(X), np.std(X), np.min(X), np.max(X), stats.skew(X), stats.kurtosis(X)]
        niu = np.mean(X)
        minX = np.min(X)
        maxX = np.max(X)
        niu2 = np.mean([x**2 for x in X])
        sigma = max(0, niu2 - niu**2) ** 0.5
        if sigma == 0:
            skew, kurt = 0, 0
        else:
            niu3 = np.mean([x**3 for x in X])
            niu4 = np.mean([(x-niu)**4 for x in X])
            skew = skew =(niu3-3*niu*sigma**2-niu**3)/(sigma**3)
            kurt=niu4/(sigma**4)-3
        return [niu, minX, maxX, skew, kurt]
        
    
    def _update_frame(self, frame_idx:int):
        frame = self.frames[frame_idx]
        res = None
        total_force = 0
        total_area = 0
        total_inten = 0
        for contact in frame.contacts:
            if contact.state == 1:  # state == 1: start frame of a contact
                self.start_frame[contact.id] = frame_idx + 1
                res = frame
            total_force += contact.force
            total_area += contact.area
            total_inten += contact.force / contact.area
        
        self.total_forces.append(total_force)
        self.total_areas.append(total_area)
        self.total_intens.append(total_inten)
        self.updated_frame_cnt = len(self.total_forces)
        return res
    
    
    def _get_contact(self, t, id):
        st = t
        en = t - 1
        # find the earliest contact index with id == id, name it st
        while st > 0:
            key_contact = None
            # search contact in one frame before
            for contact in self.frames[st-1].contacts:
                if contact.id == id:
                    key_contact = contact
                    break
            # if last frame has no contact with id == id, then st found
            if key_contact == None: break
            st -= 1                             # try one ealier frame
            if key_contact.state == 1: break    # reach the start of a contact, st found

        while en+1 < self.updated_frame_cnt:
            key_contact = None
            for contact in self.frames[en+1].contacts:
                if contact.id == id:
                    key_contact = contact
                    break
            if key_contact == None: break
            en += 1
            if key_contact.state == 3: break

        contacts = []
        
        for t in range(st, en+1):
            key_contact = None
            for contact in self.frames[t].contacts:
                if contact.id == id:
                    key_contact = contact
                    break
            contacts.append(key_contact)
        return st, en, contacts
    

    def _complete_taps(self, taps, length):
        no_tap = [0, 0, 0, 0, 0, 0]
        if len(taps) > length:
            taps = taps[:length]
        while len(taps) < length:
            taps.append(no_tap)
        for i in range(len(taps)):
            taps[i] = taps[i][:-1]
        return taps
    

    def _get_feature(self, id):
        st, en, contacts = self._get_contact(self.updated_frame_cnt-1, id)

        for contact in contacts:
            if contact.major == 0:
                return []

        feature = []

        areas = [contact.area for contact in contacts]
        ells = [float(contact.minor) / contact.major for contact in contacts]
        displacement = [((contact.x-contacts[0].x)**2+(contact.y-contacts[0].y)**2)**0.5 for contact in contacts]
        feature += self._get_sequence(areas) + self._get_sequence(ells) + self._get_sequence(displacement)

        forces = [contact.force for contact in contacts]
        intens = [contact.force / contact.area for contact in contacts]
        feature += self._get_sequence(forces) + self._get_sequence(intens) 

        frac_areas = [contacts[i].area / self.total_areas[st+i] for i in range(len(contacts))]
        frac_forces = [contacts[i].force / self.total_forces[st+i] for i in range(len(contacts))]
        frac_intens = [(contacts[i].force / contacts[i].area) / self.total_intens[st+i] for i in range(len(contacts))]
        feature += self._get_sequence(frac_areas) + self._get_sequence(frac_forces) + self._get_sequence(frac_intens)
        
        dist2edges = [min(1-contact.y,min(contact.x, 1-contact.x)) for contact in contacts]
        dist2corners = [min(contact.x**2+(1-contact.y)**2, (1-contact.x)**2+(1-contact.y)**2)**0.5 for contact in contacts]
        feature += self._get_sequence(dist2edges) + self._get_sequence(dist2corners)

        other_taps = []
        for t in range(en-1, max(st-int(5*self.FPS), 0)-1, -1):
            for contact in self.frames[t].contacts:
                if contact.state == 1 and not (t == st and contact.id == id) and len(other_taps) < 10: # whether the start of another contact
                    S, E, C = self._get_contact(t, contact.id)
                    st_time = float(S - st) / self.FPS
                    dist = np.mean([((c.x-contacts[-1].x)**2+(c.y-contacts[-1].y)**2)**0.5 for c in C])
                    force = np.mean([c.force for c in C])/np.mean(forces)
                    area = np.mean([c.area for c in C])/np.mean(areas)
                    inten = np.mean([c.force / c.area for c in C])/np.mean(intens)
                    en_time = float(E - st) / self.FPS
                    other_taps.append([st_time, dist, force, area, inten, en_time])
        pre_taps = [tap for tap in other_taps if tap[0] < 0] # Add 5 pre taps
        pre_taps.reverse()
        feature.extend(np.array(self._complete_taps(pre_taps, 5)).flatten())
        close_taps = [tap for tap in other_taps if tap[-1] >= 0] # Add 5 closest taps during this tap
        close_taps.sort(key=lambda tap: tap[1])
        feature.extend(np.array(self._complete_taps(close_taps, 5)).flatten())
        return feature
    
    
    def _get_key_contact(self, frame_idx:int): # Return contacts which are right to judge (5 frames or the end).
        # Each 'contact' add member variables 'feature', 'duration'
        D1, D2 = 5, 5 # in frames
        contacts = []

        frame = self.frames[frame_idx]
        for contact in frame.contacts:
            duration = self.updated_frame_cnt - self.start_frame[contact.id]
            # filter out front 5 frames at most in a single contact to judge
            if (D1 <= duration and duration <= D2) or (duration < D1 and contact.state == 3):
                feature = self._get_feature(contact.id)
                if len(feature) != 0:
                    contact.duration = duration
                    contact.feature = feature
                    contacts.append(contact)

        return contacts
        
    
    def close(self):
        self.is_running = False


    def get_frame(self):
        while (len(self.frames) == 0):
            time.sleep(0.001)
        return self.frames[-1]
    
    
    def get_new_frames(self) -> List[FrameData]:
        while self.updated == False:
            time.sleep(0.001)
                        
        length = len(self.frames)
        if self.label_contact:
            for frame_idx in range(self.frame_cnt, length):
                self._update_frame(frame_idx)
                key_contacts = self._get_key_contact(frame_idx)
                if len(key_contacts) > 0:
                    features = [contact.feature for contact in key_contacts]
                    features = self.scalar.transform(features)
                    preds = self.clf.predict(features)
                    for i, contact in enumerate(key_contacts):
                        if preds[i]: contact.label = 1
                        else: contact.label = 0
                        
        self.updated = False
        res = self.frames[self.frame_cnt:length]
        self.frame_cnt = length        
        return res
    
    
    def get_frame_time(self):
        if len(self.frames) >= 2:
            return round(self.frames[-1].timestamp - self.frames[-2].timestamp, 5)
        return 0
    
    
def broadcast_contact_labels(all_contacts:list):
    contact_ids = set()
    for i, contacts in enumerate(all_contacts):
        for c1 in contacts:
            if c1.state == 1:
                contact_ids.add(c1.id)
            if c1.id not in contact_ids:
                continue
            if c1.label != -1:
                if c1.state != 1:
                    j = i - 1
                    while j >= 0:
                        c2 = None
                        for c in all_contacts[j]:
                            if c.id == c1.id:
                                c2 = c
                                break
                        if c2 is not None:
                            c2.label = c1.label
                        if c2 is None or c2.state == 1:
                            break
                        j -= 1
                if c1.state != 3:
                    j = i + 1
                    while j < len(all_contacts):
                        c2 = None
                        for c in all_contacts[j]:
                            if c.id == c1.id:
                                c2 = c
                                break
                        if c2 is not None:
                            c2.label = c1.label
                        if c2 is None or c2.state == 3:
                            break
                        j += 1
                contact_ids.remove(c1.id)
    return all_contacts
