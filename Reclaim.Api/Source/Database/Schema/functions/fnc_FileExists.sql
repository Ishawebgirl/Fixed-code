SET QUOTED_IDENTIFIER ON 
GO
SET ANSI_NULLS ON 
GO
CREATE FUNCTION [dbo].[fnc_FileExists] (
  @filename VARCHAR(265))
RETURNS INT
AS
BEGIN

-- https://www.sqlservercentral.com/scripts/sql-server-procedure-to-script-tables


/*
SQLServer User-Defined Function

PURPOSE
Check if file exists.
The function returns 0 (file does not exist), or 1 (file exists).

EXAMPLES
SELECT dbo.fnc_FileExists('C:\oneTouch_drive.cvf')
1
SELECT dbo.fnc_FileExists('C:\xxxx.txt')
0

HISTORY
24-nov-2015 - Created by Gerrit Mantel

TAGS
<program>
  <description>Check if file exists</description>
  <generic>1</generic>
  <author>Gerrit Mantel</author>
  <created>2015-11-24</created>
  <lastmodified>2015-11-24</lastmodified>
</program>
*/  DECLARE @fso INT
  DECLARE @hr INT
  DECLARE @ofile INT
  DECLARE @result INT

  EXEC @hr = sp_OACreate 'Scripting.FileSystemObject', @fso OUT;
  EXEC @hr = sp_OAMethod @fso, 'FileExists', @ofile OUT, @filename;
  EXEC @hr = sp_OADestroy @fso;

  SET @result = @ofile;
  RETURN @result;
END
GO
SET QUOTED_IDENTIFIER OFF 
GO
SET ANSI_NULLS OFF 
GO

GO
