import numpy as np
from .library import library

settings_tmpl = """<settings>
{fields}
  <sageinput>
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

def get_settings_xml(dtype):
    fields = []
    for name, type in dtype.descr:
        if type in [np.int32, np.uint32]:
            type = 'int'
        elif type in [np.int64, np.uint64]:
            type = 'long long'
        else:
            type = 'float'
        fields.append('    <Field Type="%s">%s</Field>'%(name, type))
    fields = '\n'.join(fields)
    return settings_tmpl.format(fields=fields, box_size=library['box_size'])
