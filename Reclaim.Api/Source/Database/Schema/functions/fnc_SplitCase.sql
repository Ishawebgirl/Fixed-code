SET QUOTED_IDENTIFIER ON 
GO
SET ANSI_NULLS ON 
GO
CREATE FUNCTION [dbo].[fnc_SplitCase] (@input nvarchar(4000))
returns varchar(4000)
as begin
	
	if @input is null return null

	declare @char as nchar(1)
	declare @i as int = 0
	declare @output as nvarchar(4000) = ''

	while @i <= len(@input) begin	
		set @i = @i + 1
		set @char = substring(@input, @i, 1)
		
		if @i > 1 and ascii(@char) between 65 and 90
			set @output = @output + ' ' + lower(@char)
		else
			set @output = @output + @char		
	end

	return @output

end
GO
SET QUOTED_IDENTIFIER OFF 
GO
SET ANSI_NULLS OFF 
GO

GO
