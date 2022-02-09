import simplestereo as ss

# Build a SVG chessboard for calibration

# Internal (columns, rows) of intersection points of the chessboard
dimensions = (7,6)      
# Export file path
path = 'chessboard.svg' 
# Side of the square in millimeters
squareSize = 50         

# Note that being a vector image it can be scaled losslessly
ss.calibration.generateChessboardSVG(dimensions, path, squareSize)

print("Done!")
