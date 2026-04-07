import numpy as np
from .board import Board


class History():
    ''' record of past frames for real-time unintentional touches recognition
    attrs:
        N: max number of contacts in one frame (UNSURE)
        frames: list of past frames
        start_frame: TODO
        total_forces: TODO
        total_intens: TODO
    '''
    
    def __init__(self):
        self.N = 16
        self.frames = []
        self.start_frame = np.zeros(self.N, dtype=np.int32)
        self.total_forces = []
        self.total_areas = []
        self.total_intens = []
        self.spilled_frames = 0


    def _get_sequence(self, X):
        # a Faster Implementation of: return [np.mean(X), np.std(X), np.min(X), np.max(X), stats.skew(X), stats.kurtosis(X)]
        niu = np.mean(X)
        minX = np.min(X)
        maxX = np.max(X)
        niu2 = np.mean([x**2 for x in X])
        sigma = max(0, niu2 - niu**2) ** 0.5
        if sigma == 0:
            skew = 0
            kurt = 0
        else:
            niu3 = np.mean([x**3 for x in X])
            niu4 = np.mean([(x-niu)**4 for x in X])
            skew = skew =(niu3-3*niu*sigma**2-niu**3)/(sigma**3)
            kurt=niu4/(sigma**4)-3
        return [niu, minX, maxX, skew, kurt]


    def _check_spill(self):
        max_frames = 1100 # save max_frames frames in memory at most
        min_frames = 100  # frames needed for history
        len_frames = len(self.frames)
        if len_frames >= max_frames:
            self.spilled_frames += (len_frames - min_frames)
            self.frames = self.frames[-min_frames:]
            

    def update_frame(self, frame):
        self._check_spill()
        res = None
        self.frames.append(frame)

        total_force = 0
        total_area = 0
        total_inten = 0
        for contact in frame.contacts:
            if contact.state == 1:  # state == 1: start frame of a contact
                self.start_frame[contact.id] = self.spilled_frames + len(self.frames)
                res = frame
            total_force += contact.force
            total_area += contact.area
            total_inten += contact.force / contact.area
        
        self.total_forces.append(total_force)
        self.total_areas.append(total_area)
        self.total_intens.append(total_inten)
        return res


    def get_contact(self, t, id):
        st = t
        en = t - 1
        # find the earliest contact index with id == id, name it st
        while (st > self.spilled_frames):
            key_contact = None
            # search contact in one frame before
            for contact in self.frames[st - 1 - self.spilled_frames].contacts:
                if contact.id == id:
                    key_contact = contact
                    break
            # if last frame has no contact with id == id, then st found
            # TODO think why these two break branch can not merge
            if key_contact == None: break
            # try one ealier frame
            st -= 1
            # reach the start of a contact, st found
            if key_contact.state == 1: break

        while (en + 1 < len(self.frames) + self.spilled_frames):
            key_contact = None
            for contact in self.frames[en + 1 - self.spilled_frames].contacts:
                if contact.id == id:
                    key_contact = contact
                    break
            if key_contact == None: break
            en += 1
            if key_contact.state == 3: break

        contacts = []
        for t in range(st, en + 1):
            key_contact = None
            for contact in self.frames[t - self.spilled_frames].contacts:
                if contact.id == id:
                    key_contact = contact
                    break
            contacts.append(key_contact)
        return st, en, contacts


    def complete_taps(self, taps, length):
        no_tap = [0, 0, 0, 0, 0, 0]
        if len(taps) > length:
            taps = taps[:length]
        while len(taps) < length:
            taps.append(no_tap)
        for i in range(len(taps)):
            taps[i] = taps[i][:-1]
        return taps


    def get_feature(self, id):
        st, en, contacts = self.get_contact(len(self.frames)+self.spilled_frames-1, id)

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
        for t in range(en-1, max(st-int(5*Board.FPS), self.spilled_frames)-1, -1):
            for contact in self.frames[t - self.spilled_frames].contacts:
                if contact.state == 1 and not (t == st and contact.id == id) and len(other_taps) < 10: # whether the start of another contact
                    S, E, C = self.get_contact(t, contact.id)
                    st_time = float(S - st) / Board.FPS
                    dist = np.mean([((c.x-contacts[-1].x)**2+(c.y-contacts[-1].y)**2)**0.5 for c in C])
                    force = np.mean([c.force for c in C])/np.mean(forces)
                    area = np.mean([c.area for c in C])/np.mean(areas)
                    inten = np.mean([c.force / c.area for c in C])/np.mean(intens)
                    en_time = float(E - st) / Board.FPS
                    other_taps.append([st_time, dist, force, area, inten, en_time])
        pre_taps = [tap for tap in other_taps if tap[0] < 0] # Add 5 pre taps
        pre_taps.reverse()
        feature.extend(np.array(self.complete_taps(pre_taps, 5)).flatten())
        close_taps = [tap for tap in other_taps if tap[-1] >= 0] # Add 5 closest taps during this tap
        close_taps.sort(key=lambda tap: tap[1])
        feature.extend(np.array(self.complete_taps(close_taps, 5)).flatten())
        return feature
    
    
    def get_key_contact(self, frame): # Return contacts which are right to judge (5 frames or the end).
        # Each 'contact' add member variables 'feature', 'duration'
        D1, D2 = 5, 5 # in frames
        contacts = []

        for contact in frame.contacts:
            duration = len(self.frames) + self.spilled_frames - self.start_frame[contact.id]
            # filter out front 5 frames at most in a single contact to judge
            if (D1 <= duration and duration <= D2) or (duration < D1 and contact.state == 3):
                feature = self.get_feature(contact.id)
                if len(feature) != 0:
                    contact.duration = duration
                    contact.feature = feature
                    contacts.append(contact)

        return contacts
