NAME          FIXSPACE
ROWS
 G  R FIRST 
 N  COST ONE
 L  R RANGE 
 E  R EQUAL 
COLUMNS
    X ONE     COST ONE             1
    X ONE     R FIRST              1
    X ONE     R RANGE              1
    Y TWO     COST ONE             2
    Y TWO     R FIRST              1
    Y TWO     R RANGE             -1
    Z FIXED   COST ONE             3
    Z FIXED   R FIRST              1
    Z FIXED   R EQUAL              1
RHS
    RHS ONE   R FIRST              4
    RHS ONE   COST ONE            -7
    RHS ONE   R RANGE              2
    RHS ONE   R EQUAL              1
RANGES
    RNG ONE   R RANGE              4
BOUNDS
 UP BND ONE   X ONE                4
 FR BND ONE   Y TWO   
 FX BND ONE   Z FIXED              1
ENDATA
