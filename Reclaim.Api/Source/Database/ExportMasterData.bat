
set outputpath=z:\Dev\Github\Reclaim.Api\Source\Database\MasterData

for %%X in (
	ChatRole
	ChatType
	ClaimDisposition
    	ClaimType
    	ClaimStatus
	CustomerStatus
	DocumentType
	EmailStatus
	EmailTemplate
	ErrorCode
	IdentityProvider
	InvestigatorStatus
	JobStatus
	JobType
	LogEntryLevel
	OwnershipType
	PropertyType
	Role
	RoofType
	State
) do (sqlcmd -y 0 -Q "exec util_GenerateInserts '%%X'" -d Reclaim -o %outputpath%\%%X.txt)
