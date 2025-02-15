import { useEffect, useMemo, useState, useRef } from 'react'
import Table from 'components/Table'
import * as Model from 'api/model'
import FilterTableMenuItem from 'components/FilterTableMenuItem'
import ExportTableMenuItem from 'components/ExportTableMenuItem'
import Menu from 'components/Menu'
import { Log } from 'helpers/log'

const SignIns = () => {
  const accountApi = useMemo(
    () => new Model.AccountClient(process.env.REACT_APP_API_URL),
    []
  )

  const tableRef = useRef<{ handleExport: () => void } | null>(null)

  const [accounts, setAccounts] = useState<Model.Account[]>()
  const [filterTerms, setFilterTerms] = useState('')

  const columns = useMemo(
    () => [
      {
        label: '',
        accessor: 'avatarUrl',
        type: 'avatar',
      },
      {
        label: 'Name / Email',
        accessor: 'niceName',
        type: 'niceNameAndEmailAddress',
      },
      {
        label: 'Role',
        accessor: 'role',
        type: 'propertyTag',
      },
      {
        label: 'Signed in',
        accessor: 'sessionAuthenticatedTimestamp',
        type: 'timeAgo',
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
        const result = await accountApi.authenticated()
        setAccounts(result)
      } catch (error) {
        Log.add(JSON.stringify(error))
      }
    })()
  }, [accountApi])

  return (
    accounts && (
      <>
        <div className="header">Sign-ins</div>
        <Menu>
          {accounts.length > 0 ? (
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
          id="sign-in-table"
          name="Sign-Ins"
          keyField="emailAddress"
          columns={columns}
          sourceData={accounts}
          isPropertyBarVisible={false}
          filterTerms={filterTerms}
          onRowClick={null}
          initialSortColumn={'lastActiveTimestamp'}
        />
      </>
    )
  )
}

export default SignIns
