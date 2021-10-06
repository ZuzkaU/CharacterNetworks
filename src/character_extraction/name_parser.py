import os
import json

class Person:
    def __init__(self, gender, honorific, first_name, last_name, name_variants=[]):
        self.gender = gender
        self.honorific = honorific
        self.first_name = first_name
        self.last_name = last_name
        self.name_variants = name_variants
    def infoString(self):
        return ('PERSON: \n\tgender {}\n\thonorific {}\n\tfirst name {}\n\tlast_name {}\n\t\n').format(self.gender, self.honorific, self.first_name, self.last_name)
    def __str__(self):
        return ' '.join(filter(None, [self.honorific, self.first_name, self.last_name]))
        
    def honorificDiffer(self, other):
        if self.honorific and other.honorific and not self.honorific == other.honorific:
            return True
        return False
    
    def isSubsetOf(self, other):
        # self subset of other
        if self.honorific and not self.honorific == other.honorific:
            return False
        if self.first_name and not self.first_name == other.first_name:
            return False
        if self.last_name and not self.last_name == other.last_name:
            return False
        return True
        
    def firstNamesVariant(self, other):
        if not self.first_name or not other.first_name:
            return False
        if self.first_name in other.name_variants or other.first_name in self.name_variants:
            return True
        if self.first_name == other.first_name:
            return True
        return False
            
    def namePartDiffer(self, other):
        if self.first_name and other.first_name and not self.first_name == other.first_name:
            return True
        if self.last_name and other.last_name and not self.last_name == other.last_name:
            return True
        return False


class NameParser:
    def __init__(self, names):
        self.names = names
        vocab_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'vocab')
        
        with open(os.path.join(vocab_dir, 'honorific.json')) as honorific_file:
            self.honorific = json.load(honorific_file)
        
        with open(os.path.join(vocab_dir, 'female.txt')) as woman_file:
            names = [name for name in woman_file.read().splitlines() if name and not name.startswith('#')]
            self.names_woman = set(names)
        with open(os.path.join(vocab_dir, 'male.txt')) as man_file:
            names = [name for name in man_file.read().splitlines() if name and not name.startswith('#')]
            self.names_man = set(names)
        
        with open(os.path.join(vocab_dir, 'hypocorisms.txt')) as variants_file:
            variants = {}
            for line in variants_file.read().splitlines():
                names = line.split()
                variants[names[0]] = names[1:]
            self.variants = variants
        
        self.first_names, self.last_names = [], []
    
    
    def accumulateFirstAndLastNames(self):
        self.first_names = set()
        self.last_names = set()
        for full_name in self.names:
            honor, name = self.splitHonorName(full_name)
            parts = name.split()
            if len(parts) >= 2:
                _, first_name, last_name = self.parseName(full_name)
                self.first_names.add(first_name)
                self.last_names.add(last_name)
        return
    
    def parseNames(self):
        self.dict = {}
        
        self.accumulateFirstAndLastNames()
        
        for name in self.names:
            (honor, first, last) = self.parseName(name)
            gender = self.getGender(honor, first)
            if first in self.variants:
                varnames = self.variants[first]
            else:
                varnames = []
            self.dict[name] = Person(gender, honor, first, last, varnames)
        
        return self.dict

    def parseName(self, full_name):
        honor, name = self.splitHonorName(full_name)
        parts = name.split()
        first_name, last_name = None, None
        
        if len(parts) == 1:
            if parts[0] in self.last_names:
                last_name = parts[0]
            elif parts[0] in self.first_names:
                first_name = parts[0]
            elif self.getGender(honor, None) == 'F' and parts[0] in self.names_woman:
                first_name = parts[0]
            elif self.getGender(honor, None) == 'M' and parts[0] in self.names_man:
                first_name = parts[0]
            else:
                last_name = parts[0]
        elif len(parts) == 2:
            if parts[0].lower in ["de", "von"]:
                last_name = name
            else:
                last_name = parts[1]
                first_name = parts[0]
        elif len(parts) == 3:
            first_name = parts[0]
            last_name = ' '.join(parts[1:])
        else:
            first_name = ' '.join(parts[0:2])
            last_name = ' '.join(parts[2:])
        
        return (honor, first_name, last_name)

    def getGender(self, honor, first_name):
        if honor:
            if honor in self.honorific["woman"]:
                return 'F'
            elif honor in self.honorific["man"]:
                return 'M'
        if first_name:
            if first_name in self.names_man:
                return 'M'
            elif first_name in self.names_woman:
                return 'F'
        return None
    
    def splitHonorName(self, name):
        parts = name.split()
        honors = self.honorific["woman"] + self.honorific["man"] + self.honorific["other"]
        if parts[0] in honors:
            return parts[0], ' '.join(parts[1:])
        return None, name

