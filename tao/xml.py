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
    <Version>
    {dataset-version}
    </Version>
  </PGDB>
  <RunningSettings>
    <Simulation>{sim_name}</Simulation>
    <GalaxyModel>{model_name}</GalaxyModel>
    <hubble>{hubble}</hubble>
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
    <item>globalindex</item>
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


def sanitize_string(field):
    new_field = field
    invalid_to_valid_xml = {'&':'&amp;'}
    for k in invalid_to_valid_xml.keys():
        v = invalid_to_valid_xml[k]
        try:
            if k in new_field:
                new_field = str.replace(new_field, k, v)
        except TypeError:
            pass
        

    return new_field    

def get_settings_xml(dtype, redshifts, metadata):

    fields = []
    wanted_metadata_keys  = ['label', 'description', 'order',
                               'units', 'group']
    all_orders = []
    all_orders_names = []
    for name in dtype.names:
        d = retrieve_dict_from_metadata(name, metadata)
        try:
            val = d['order']
            assert isinstance( val, ( int, long ) ),\
                "val = {0} is not an integer".format(val)
            
            all_orders.append(val)
            all_orders_names.append(name)
            
        except KeyError:
            pass     
            
    uniq_orders =  np.unique(all_orders != -1)
    assert len(uniq_orders) == len((np.where(all_orders != -1))[0]),\
        "Orders are not unique "\
        "orders = {0}".format(all_orders)

    # Fix there are holes in the order
    non_negative_ind = [i for i,j in enumerate(all_orders) if j >= 0]
    valid_orders = [all_orders[i] for i in non_negative_ind]
    valid_names  = [all_orders_names[i] for i in non_negative_ind]
    sorted_ind = np.argsort(valid_orders)

    sorted_orders = [valid_orders[i] for i in sorted_ind]
    sorted_names  = [valid_names[i] for i in sorted_ind]
    
    for new_idx, name in enumerate(sorted_names):
        d = retrieve_dict_from_metadata(name, metadata)
        val = d['order']
        assert val >= 0, "Bug in tao: order should not have been -1 here"
        
        d['order'] = new_idx
        # print("Name = {2} old = {0} to new = {1}".format(val, d['order'], name))
    
    # Okay the orders are unique and do not have holes.
    # Find the max and assign orders to the fields that
    # do not already have an order.
    order_index = len(sorted_names) 
            
    for name, typ in dtype.descr:
        typ = np.dtype(typ)
        if typ in [np.int32, np.uint32]:
            typ = 'int'
        elif typ in [np.int16, np.uint16]:
            typ = 'short'
        elif typ in [np.int64, np.uint64]:
            typ = 'long long'
        else:
            typ = 'float'

        this_field = '    <Field Type="{0}"'.format(typ)
        d = retrieve_dict_from_metadata(name, metadata)
        for key in wanted_metadata_keys:
            try:
                val = d[key]
                if key == 'order' and val < 0:
                    val = order_index
                    order_index += 1
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
            val = sanitize_string(val)
            this_field = '{0}\n        {1}="{2}"'.format(this_field, key, val)

        name = sanitize_string(name)

        this_field = '{0}>{1}</Field>'.format(this_field, name)
        fields.append(this_field)
    fields = '\n'.join(fields)

    snapshots = []
    for z in redshifts[::-1]:
        snapshots.append('      <Snapshot>%s</Snapshot>'%(z))

    snapshots = '\n'.join(snapshots)
    return settings_tmpl.format(fields=fields,
                                redshifts=snapshots,
                                sim_name = library['sim-name'],
                                model_name = library['model-name'],
                                hubble = library['hubble'],
                                box_size=library['box_size'],
                                dataset-version=library['dataset-version'])
