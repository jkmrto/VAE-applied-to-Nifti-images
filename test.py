import numpy as np
import json





out1 = np.array([1,2,3,4,5,6,7,7,8])
out2 = np.array([8,5,3,4,5,6,7,7,8])

out = {
    "out1": out1,
    "out2":  out2,
}


out = {
    1:out,
    2:out,
    3:out,
}


file = open("test.json", "w")
json.dump(out, file, cls=JSONEncoder)