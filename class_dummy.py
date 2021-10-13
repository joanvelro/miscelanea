import pandas
import numpy

data = pandas.DataFrame({'Purchase Cost': [23, 45, 67, 78],
                         'name':['product-A','product-B','product-C','product-D'],
                         'Lead Time': [23, 45, 67, 78],
                         'Size': [23, 45, 67, 78],
                         'Selling Price': [23, 45, 67, 78],
                         'Ch': [23, 45, 67, 78],
                         'Co': [23, 45, 67, 78],
                         'Probability': [23, 45, 67, 78],
                         'Starting Stock': [23, 45, 67, 78],
                         'Demand_lead': [23, 45, 67, 78]})


class Product:
    def __init__(self, i):
        """
        :type i: int - Product number
        """
        self.i = i
        self.unit_cost = data['Purchase Cost'].iloc[i - 1]
        self.lead_time = data['Lead Time'].iloc[i - 1]
        self.size = data['Size'].iloc[i - 1]
        self.name = data['name'].iloc[i - 1]
        self.selling_price = data['Selling Price'].iloc[i - 1]
        self.holding_cost = data['Ch'].iloc[i - 1]
        self.ordering_cost = data['Co'].iloc[i - 1]
        self.probability = data['Probability'].iloc[i - 1]
        self.starting_stock = data['Starting Stock'].iloc[i - 1]
        self.demand_lead = data['Demand_lead'].iloc[i - 1]

        #self.mean = numpy.mean([numpy.log(j) for j in data[data[i] > 0][i]])
        #self.sd = numpy.std([numpy.log(j) for j in data[data[i] > 0][i]])


product = Product(1)

print(product.size)
print(product.name)