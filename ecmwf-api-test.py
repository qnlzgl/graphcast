# # ai-models
# class MarsInput(RequestBasedInput):
#     WHERE = "MARS"

#     def __init__(self, owner, **kwargs):
#         self.owner = owner

#     def pl_load_source(self, **kwargs):
#         kwargs["levtype"] = "pl"
#         logging.debug("load source mars %s", kwargs)
#         return cml.load_source("ecmwf-open-data", **kwargs)

#     def sfc_load_source(self, **kwargs):
#         kwargs["levtype"] = "sfc"
#         logging.debug("load source mars %s", kwargs)
#         return cml.load_source("ecmwf-open-data", **kwargs)

#     def ml_load_source(self, **kwargs):
#         kwargs["levtype"] = "ml"
#         logging.debug("load source mars %s", kwargs)
#         return cml.load_source("ecmwf-open-data", **kwargs)


import climetlab as cml
source = cml.load_source(
    "ecmwf-open-data",
    date="20240512",
    time="00",
    param=["lsm","t2m","msl","u10","v10","tp","z"],   # MARS symbol: ["lsm","2t","msl","10u","10v","tp","z"]
    grid=[0.25,0.25],
    area=[90,0,-90,360],
    type="fc",
    stream="oper", # scda for 6 and 18 time
    levtype="sfc",
    # target="result.grib2",
    # format="grib2"
)


MARS_TO_OPENDATA = {
    "2t": "t2m",
    "10u": "u10",
    "10v": "v10",
}

s = cml.load_source("ecmwf-open-data",
    date=20240512,
    time=18,
    param=['t', 'z', 'u', 'v', 'w', 'q'],   # MARS symbol: ["lsm","2t","msl","10u","10v","tp","z"]
    level=[50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000], 

    grid=[0.25,0.25],
    area=[90,0,-90,360],
    type="fc",
    stream="scda", # scda for 6 and 18 time
    levtype="pl"
)




# a = source.to_xarray()
# for s in source:
#     cml.plot_map(s)

import re
import sys
from ai_models.__main__ import main
if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    main()



# from ecmwfapi import ECMWFDataServer

# # To run this example, you need an API key
# # available from https://api.ecmwf.int/v1/key/

# server = ECMWFDataServer()
# server.retrieve({
#     'origin'    : "ecmf",
#     'levtype'   : "sfc",
#     'number'    : "1",
#     'expver'    : "prod",
#     'dataset'   : "tigge",
#     'step'      : "0/6/12/18",
#     'area'      : "70/-130/30/-60",
#     'grid'      : "2/2",
#     'param'     : "167",
#     'time'      : "00/12",
#     'date'      : "2014-11-01",
#     'type'      : "pf",
#     'class'     : "ti",
#     'target'    : "tigge_2014-11-01_0012.grib"
# })

# from ecmwf.opendata import Client

# client = Client(source="ecmwf", model="ifs", resol="0p25")

# client.retrieve(
#     time=0,
#     type="fc",
#     stream="oper",
#     step=0,
#     # param="2t",
#     target="data.grib2",
#     date='2024-05-11'
# )

# from ecmwf.opendata import Client
# import ecmwf.data as ecdata

# client = Client(source="ecmwf", model="ifs", resol="0p25")
# # Specify the parameters for the request
# request = {
#     "date": "20240511",
#     "time": "00",
#     "param": ["lsm","2t","msl","10u","10v","tp","z"],
#     "grid": [0.25,0.25],
#     "area": [90,0,-90,360],
#     "type": "fc",
#     "stream": "oper", # scda for 6 and 18 time
#     "levtype": "sfc",
#     "target": "result.grib2",
#     "format": "grib2"
# }
# client.retrieve(**request)

# # Define the output path for the GRIB file
# output_file = '20240511000000-0h-oper-fc.grib2'

# # Retrieve the data
# with open(output_file, "wb") as file:
#     client.retrieve(request, file)

# print(f"File downloaded: {output_file}")


# {
#   "date": 20230110,
#   "time": 0,

#   "grid": [
#     0.25,
#     0.25
#   ],
#   "area": [
#     90,
#     0,
#     -90,
#     360
#   ],
#   "type": "fc",
#   "stream": "oper"
# }