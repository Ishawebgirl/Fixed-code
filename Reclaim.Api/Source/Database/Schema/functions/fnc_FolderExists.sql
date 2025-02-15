SET QUOTED_IDENTIFIER ON 
GO
SET ANSI_NULLS ON 
GO
CREATE FUNCTION [dbo].[fnc_FolderExists] (
  @dirname VARCHAR(265))
RETURNS INT
AS
BEGIN

-- https://www.sqlservercentral.com/scripts/sql-server-procedure-to-script-tables

/*
SQLServer User-Defined Function

PURPOSE
Check if folder exists.
The function returns 0 (folder does not exist), or 1 (folder exists).

EXAMPLES
SELECT dbo.fnc_FolderExist('C:\Program Files')
1
SELECT dbo.fnc_FolderExist('C:\xxxx')
0

HISTORY
24-nov-2015 - Created by Gerrit Mantel

TAGS
<program>
  <description>Check if folder exists</description>
  <generic>1</generic>
  <author>Gerrit Mantel</author>
  <created>2015-11-24</created>
  <lastmodified>2015-11-24</lastmodified>
</program>
*/  DECLARE @fso INT
  DECLARE @hr INT
  DECLARE @ofolder INT
  DECLARE @name VARCHAR(265)
  DECLARE @result INT

  EXEC @hr = sp_OACreate 'Scripting.FileSystemObject', @fso OUT;
  EXEC @hr = sp_OAMethod @fso, 'FolderExists', @ofolder OUT, @dirname;
  EXEC @hr = sp_OADestroy @fso;

  SET @result = @ofolder;
  RETURN @result;
END
GO
SET QUOTED_IDENTIFIER OFF 
GO
SET ANSI_NULLS OFF 
GO

GO
