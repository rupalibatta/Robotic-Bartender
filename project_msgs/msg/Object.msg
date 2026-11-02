# Types of objects that can be detected on the table.
# Constants are all UPPERCASE and equal to a value.
int32 DISK=0
int32 STRIP=1
int32 BOTTLE_BLUE=2
int32 BOTTLE_PURPLE=3

# Actual data (to be filled in when instantiated).
int32 type
geometry_msgs/Pose pose