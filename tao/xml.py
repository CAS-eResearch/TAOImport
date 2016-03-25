import numpy as np
from .library import library
# from IPython.core.debugger import Tracer

settings_tmpl = """<settings>
  <sageinput>
{fields}
  </sageinput>
  <PGDB>
    <ScienceModuleDBUserName></ScienceModuleDBUserName>
    <TreeTablePrefix>tree_</TreeTablePrefix>
    <NewDBName></NewDBName>
    <NewDBAlias></NewDBAlias>
    <ServersCount>3</ServersCount>
    <serverInfo>
      <serverip></serverip>
      <port></port>
      <user></user>
      <password></password>
    </serverInfo>
    <serverInfo>
      <serverip></serverip>
      <port></port>
      <user></user>
      <password></password>
    </serverInfo>
    <serverInfo>
      <serverip></serverip>
      <port></port>
      <user></user>
      <password></password>
    </serverInfo>
    <Snapshots>
{redshifts}
    </Snapshots>
  </PGDB>
  <RunningSettings>
    <SnpshottoRedshiftMapping></SnpshottoRedshiftMapping>		
    <InputFile></InputFile>
    <SimulationBoxX>{box_size}</SimulationBoxX>
    <SimulationBoxY>{box_size}</SimulationBoxY>
    <BSPCellSize>10</BSPCellSize>		
    <GalaxiesPerTable>500000</GalaxiesPerTable>
    <OverWriteDB>yes</OverWriteDB>
    <RegenerateFilesList>yes</RegenerateFilesList>
    <RegenerateTables>yes</RegenerateTables>
  </RunningSettings>
  <TreeTraversal>
    <item>global_index</item>
    <item>descendant</item>
    <item>snapshot</item>
  </TreeTraversal>
</settings>
"""
def retrieve_dict_from_metadata(name, metadata):
    try:
        d = metadata[name]
    except KeyError:
        # May be it is a upper-case/lowercase
        try:
            d = metadata[name.lower()]
        except KeyError:
            msg = "Could not locate key = {0} "\
                "Even after trying lower-case = {1}"\
                .format(name, name.lower())
            raise KeyError(msg)

    return d


def get_settings_xml(dtype, redshifts, metadata):
    fields = []
    wanted_metadata_keys  = ['label', 'description', 'order',
                               'units', 'group']
    # import pprint
    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(metadata)
    all_orders = []
    for name, _ in dtype.descr:
        d = retrieve_dict_from_metadata(name, metadata)
        try:
            val = d['order']
            assert isinstance( val, ( int, long ) ),\
                "val = {0} is not an integer".format(val)
            
            all_orders.append(val)
        except KeyError:
            pass     
            
    uniq_orders =  np.unique(all_orders)
    assert len(uniq_orders) == len(all_orders),\
        "Orders are not unique "\
        "orders = {0}".format(all_orders)
    
    # Okay the orders are unique. Find the max and assign orders
    # to the fields that do not already have an order.
    order_index = np.max(all_orders) + 1
            
    for name, typ in dtype.descr:
        typ = np.dtype(typ)
        if typ in [np.int32, np.uint32]:
            typ = 'int'
        elif typ in [np.int64, np.uint64]:
            typ = 'long long'
        else:
            typ = 'float'

        this_field = '    <Field Type="{0}"'.format(typ)
        d = retrieve_dict_from_metadata(name, metadata)
        for key in wanted_metadata_keys:
            try:
                val = d[key]
            except KeyError:
                if key == 'order':
                    val = order_index
                    order_index += 1
                elif key == 'label':
                    val = name
                elif key == 'description':
                    val = name
                elif key == 'group':
                    val = "Internal"
                elif key == 'units':
                    val = ""

                # print "HAD KEYERROR - passing (key, val) = {0}, {1} for name = {2}"\
                #     .format(key, val, name)
            
            this_field = '{0}\n        {1}="{2}"'.format(this_field, key, val)
                
        this_field = '{0}>{1}</Field>'.format(this_field, name)
        fields.append(this_field)
    fields = '\n'.join(fields)

    snapshots = []
    for z in redshifts[::-1]:
        snapshots.append('      <Snapshot>%s</Snapshot>'%(z))

    snapshots = '\n'.join(snapshots)
    return settings_tmpl.format(fields=fields,
                                redshifts=snapshots,
                                box_size=library['box_size'])
