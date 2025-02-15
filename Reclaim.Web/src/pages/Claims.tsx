import { useEffect, useMemo, useRef, useState } from 'react'
import * as Model from 'api/model'
import { useNavigate } from 'react-router-dom'
import { Log } from 'helpers/log'
import Menu from 'components/Menu'
import Table from 'components/Table'
import ExportTableMenuItem from 'components/ExportTableMenuItem'
import FilterTableMenuItem from 'components/FilterTableMenuItem'

const Claims = () => {
  const columns = useMemo(
    () => [
      {
        label: 'ID / Value',
        accessor: 'externalID',
        type: 'claimExternalIDAndValue',
      },
      {
        label: 'Address',
        accessor: 'policy.address',
        type: 'fullAddress',
      },
      {
        label: 'Date',
        accessor: 'eventDate',
        type: 'date',
      },
      {
        label: 'Type',
        accessor: 'type',
      },
      {
        label: 'Status',
        accessor: 'status',
        type: 'propertyTag',
      },
      {
        label: 'Disposition',
        accessor: 'disposition',
        type: 'propertyTag',
      },
    ],
    []
  )

  const claimApi = useMemo(() => new Model.ClaimClient(process.env.REACT_APP_API_URL), [])

  const [claims, setClaims] = useState<Model.Claim[]>()
  const navigate = useRef(useNavigate())
  const tableRef = useRef<{ handleExport: () => void } | null>(null)

  const [filterTerms, setFilterTerms] = useState('')

  const handleRowClick = (clickedClaim: Model.Claim) => {
    navigate.current('/claims/' + clickedClaim.uniqueID)
  }

  useEffect(() => {
    ;(async () => {
      try {
        const result = await claimApi.getClaims()
        setClaims(result)
      } catch (error) {
        Log.add(JSON.stringify(error))
      }
    })()
  }, [claimApi])

  return (
    claims && (
      <>
        <div className="header">Claims</div>
        <Menu>
          {claims.length > 0 ? (
            <>
              <ExportTableMenuItem onClick={() => tableRef.current?.handleExport()} />
              <FilterTableMenuItem
                filterTerms={filterTerms}
                onChange={(terms: string) => setFilterTerms(terms)}
              />
            </>
          ) : (
            <></>
          )}
        </Menu>
        <Table
          ref={tableRef}
          id="claims-table"
          name="Claims"
          keyField="uniqueID"
          columns={columns}
          sourceData={claims}
          isPropertyBarVisible={false}
          filterTerms={filterTerms}
          ignoredExportFields={['documents']}
          onDelete={null}
          onRowClick={handleRowClick}></Table>
      </>
    )
  )
}

export default Claims
