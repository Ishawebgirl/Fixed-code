set outputpath=z:\Dev\Github\Reclaim.Api\Source\Database\Schema

schemazen script -o --server localhost --database reclaim --scriptDir %outputpath%  --onlyTypes=tables,foreign_keys,functions,procedures,views,check_constraints,defaults
