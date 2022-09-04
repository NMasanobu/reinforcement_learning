強化学習の勉強用

# Action ID
Integer from 0 to 8, representing a cell of grid.  
| 0 | 1 | 2 |  
| 3 | 4 | 5 |  
| 6 | 7 | 8 |

# State ID
9 digits string. Each digit is 0, 1 or 2.  
The index of the string represents a cell of grid (see Action ID).  
The value represents cell status:  
* 0: Empty cell
* 1: Filled by player
* 2: Filled by cpu

012010000 represents the following state:  
| * | o | x |  
| * | o | * |  
| * | * | * |
