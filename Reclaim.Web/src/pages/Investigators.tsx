import { useEffect, useMemo, useRef, useState } from 'react'
import * as Model from 'api/model'
import React from 'react'
import { Log } from 'helpers/log'
import useIdentity from 'hooks/useIdentity'
import Menu from 'components/Menu'
import MenuItem from 'components/MenuItem'
import ExportTableMenuItem from 'components/ExportTableMenuItem'
import FilterTableMenuItem from 'components/FilterTableMenuItem'
import Table from 'components/Table'
import PropertyBar from 'components/PropertyBar'
import TextInput from 'components/TextInput'
import { RegexPattern } from 'helpers/constants'
import moment from 'moment'

const Investigators = () => {
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
        label: 'Address',
        accessor: 'address',
        type: 'fullAddress',
      },
      {
        label: 'Telephone',
        accessor: 'telephone',
      },
      {
        label: 'Status',
        accessor: 'status',
        type: 'propertyTag',
      },
      {
        label: 'Last active',
        accessor: 'lastActiveTimestamp',
        type: 'timeAgo',
      },
    ],
    []
  )

  const fields = useMemo(
    () => [
      {
        label: 'First name',
        accessor: 'firstName',
        type: 'text',
        required: true,
        columnSpec: '2',
      },
      {
        label: 'Last name',
        accessor: 'lastName',
        type: 'text',
        required: true,
        columnSpec: '2-last',
      },
      {
        label: 'Email address',
        accessor: 'emailAddress',
        type: 'text',
        required: true,
      },
      {
        label: 'Address',
        accessor: 'address',
        type: 'text',
        required: true,
      },
      {
        label: 'Apartment, suite, etc.',
        accessor: 'address2',
        type: 'text',
      },
      {
        label: 'City',
        accessor: 'city',
        type: 'text',
        required: true,
      },
      {
        label: 'State',
        accessor: 'state',
        type: 'state',
        columnSpec: '2',
        required: true,
      },
      {
        label: 'Postal code',
        accessor: 'postalCode',
        type: 'text',
        regex: RegexPattern,
        columnSpec: '2-last',
        required: true,
      },
      {
        label: 'Telephone',
        accessor: 'telephone',
        type: 'telephone',
        required: true,
      },
    ],
    []
  )

  const investigatorApi = useMemo(
    () => new Model.InvestigatorClient(process.env.REACT_APP_API_URL),
    []
  )

  const identity = useIdentity()
  const tableRef = useRef<{ handleExport: () => void } | null>(null)

  const [investigators, setInvestigators] = useState<Model.Investigator[]>()
  const [isInvestigatorEditable, setIsInvestigatorEditable] = useState<boolean>(false)
  const [isPropertyBarVisible, setIsPropertyBarVisible] = useState(false)
  const [filterTerms, setFilterTerms] = useState('')
  const [editInvestigator, setEditInvestigator] = useState<Model.Investigator>(
    {} as Model.Investigator
  )

  const handleUpdateInvestigator = async (investigator: Model.Investigator) => {
    var dto = new Model.InvestigatorCreateOrUpdate()
    dto.firstName = investigator.firstName
    dto.lastName = investigator.lastName
    dto.address = investigator.address
    dto.address2 = investigator.address2
    dto.city = investigator.city
    dto.state = investigator.state
    dto.postalCode = investigator.postalCode
    dto.emailAddress = investigator.emailAddress
    dto.telephone = investigator.telephone

    if (investigator.uniqueID === undefined) {
      await investigatorApi.create(dto)
    } else {
      await investigatorApi.update(investigator.uniqueID, dto)
    }
    var investigators = await investigatorApi.getAll()

    setInvestigators(investigators)
  }

  const updateInvestigatorProperty = (prop: string, value: string, type: string) => {
    switch (type) {
      case 'date':
        try {
          const date = moment.utc(value, 'MM/DD/YYYY').format()
          ;(editInvestigator as any)[prop] = date
        } catch {}
        break

      default:
        ;(editInvestigator as any)[prop] = value
        break
    }
  }

  const handleCancel = () => {
    setIsPropertyBarVisible(false)
    setTimeout(() => {
      setEditInvestigator(new Model.Investigator())
    }, 500)
  }

  const handleRowClick = (clickedInvestigator: Model.Investigator) => {
    setEditInvestigator(Object.assign(new Model.Investigator(), clickedInvestigator))
    setIsPropertyBarVisible(true)
  }

  useEffect(() => {
    ;(async () => {
      try {
        const result = await investigatorApi.getAll()
        setInvestigators(result)
      } catch (error) {
        Log.add(JSON.stringify(error))
      }
    })()
  }, [investigatorApi])

  useEffect(() => {
    if (identity) {
      setIsInvestigatorEditable(identity.isAdministrator())
    }
  }, [identity])

  return (
    investigators && (
      <>
        <div className="header">Investigators</div>
        <Menu>
          {isInvestigatorEditable ? (
            <MenuItem
              label="Add investigator"
              icon="SquarePlus"
              onClick={() => {
                setEditInvestigator(new Model.Investigator())
                setIsPropertyBarVisible(true)
              }}
            />
          ) : (
            <></>
          )}
          {investigators.length > 0 ? (
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
          id="investigator-table"
          name="Investigators"
          keyField="uniqueID"
          columns={columns}
          sourceData={investigators}
          isPropertyBarVisible={
            handleUpdateInvestigator !== undefined && isPropertyBarVisible
          }
          filterTerms={filterTerms}
          onRowClick={isInvestigatorEditable ? handleRowClick : null}
          initialSortColumn={'lastName'}></Table>

        {editInvestigator && handleUpdateInvestigator ? (
          <PropertyBar
            entityID={editInvestigator.uniqueID ?? ''}
            isVisible={isPropertyBarVisible}
            onSave={() => handleUpdateInvestigator(editInvestigator)}
            onCancel={handleCancel}>
            <>
              <div className="caption">
                {editInvestigator.uniqueID === undefined
                  ? 'New investigator'
                  : 'Edit investigator'}
              </div>
              {fields.map((o, i) => {
                return (
                  <TextInput
                    entityID={editInvestigator?.uniqueID?.toString()}
                    key={o.accessor}
                    type={o.type}
                    label={o.label}
                    name={o.accessor}
                    value={(editInvestigator as any)[o.accessor]}
                    required={o.required ?? false}
                    columnSpec={o.columnSpec}
                    onChange={(value: string) =>
                      updateInvestigatorProperty(o.accessor, value, o.type)
                    }
                  />
                )
              })}
            </>
          </PropertyBar>
        ) : (
          <></>
        )}
      </>
    )
  )
}

export default Investigators
