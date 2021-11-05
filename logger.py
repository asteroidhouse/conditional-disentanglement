import csv

class CSVLogger():
  def __init__(self, fieldnames, filename='log.csv'):
    self.filename = filename
    self.csv_file = open(filename, 'w')
    self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
    self.writer.writeheader()

  def writerow(self, row):
    self.writer.writerow(row)
    self.csv_file.flush()

  def close(self):
    self.csv_file.close()
