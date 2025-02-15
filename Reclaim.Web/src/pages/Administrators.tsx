import { useEffect, useMemo, useState, useRef } from 'react'
import Table from 'components/Table'
import * as Model from 'api/model'
import FilterTableMenuItem from 'components/FilterTableMenuItem'
import ExportTableMenuItem from 'components/ExportTableMenuItem'
import Menu from 'components/Menu'
import { Log } from 'helpers/log'

const Administrators = () => {
  const accountApi = useMemo(
    () => new Model.AccountClient(process.env.REACT_APP_API_URL),
    []
  )

  const tableRef = useRef<{ handleExport: () => void } | null>(null)

  const [filterTerms, setFilterTerms] = useState('')
  const [administrators, setAdministrators] = useState<Model.Administrator[]>()

  const columns = useMemo(
    () => [
      {
        label: '',
        accessor: 'avatarUrl',
        type: 'avatar',
      },
      {
        label: 'Name / Email',
        accessor: 'lastName',
        type: 'fullNameAndEmailAddress',
      },
      {
        label: 'Last active',
        accessor: 'lastActiveTimestamp',
        type: 'timeAgo',
      },
    ],
    []
  )

  useEffect(() => {
    ;(async () => {
      try {
        const result = await accountApi.getAdministrators()
        setAdministrators(result)
      } catch (error) {
        Log.add(JSON.stringify(error))
      }
    })()
  }, [accountApi])

  return (
    administrators && (
      <>
        <div className="header">Administrators</div>
        <Menu>
          {administrators.length > 0 ? (
            <>
              <ExportTableMenuItem onClick={() => tableRef.current?.handleExport()} />
              <FilterTableMenuItem
                filterTerms={filterTerms}
                onChange={(terms) => setFilterTerms(terms)}
              />
            </>
          ) : (
            <></>
          )}
        </Menu>
        <Table
          ref={tableRef}
          id="administrator-table"
          name="Administrators"
          keyField="lastName"
          columns={columns}
          sourceData={administrators}
          isPropertyBarVisible={false}
          filterTerms={filterTerms}
          onRowClick={null}
          initialSortColumn={'lastName'}
        />
      </>
    )
  )
}

export default Administrators
