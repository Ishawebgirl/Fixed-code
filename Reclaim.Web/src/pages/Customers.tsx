import { useEffect, useMemo, useRef, useState } from 'react'
import * as Model from 'api/model'
import { Log } from 'helpers/log'
import moment from 'moment'
import TextInput from 'components/TextInput'
import PropertyBar from 'components/PropertyBar'
import Table from 'components/Table'
import Menu from 'components/Menu'
import MenuItem from 'components/MenuItem'
import ExportTableMenuItem from 'components/ExportTableMenuItem'
import FilterTableMenuItem from 'components/FilterTableMenuItem'
import { RegexPattern } from 'helpers/constants'
import useIdentity from 'hooks/useIdentity'

const Customers = () => {
  const columns = useMemo(
    () => [
      {
        label: 'Name / Code',
        accessor: 'name',
        type: 'customerNameAndCode',
      },
      {
        label: 'Contact / Email',
        accessor: 'lastName',
        type: 'fullNameAndEmailAddress',
      },
      {
        label: 'Address',
        accessor: 'address',
        type: 'fullAddress',
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
        label: 'Name',
        accessor: 'name',
        type: 'text',
        required: true,
      },
      {
        label: 'Code',
        accessor: 'code',
        type: 'text',
        required: true,
        formatting: 'uppercase',
      },
      {
        label: 'Contact first name',
        accessor: 'firstName',
        type: 'text',
        required: true,
        columnSpec: '2',
      },
      {
        label: 'Contact last name',
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
        regex: RegexPattern.postalCode,
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

  const customerApi = useMemo(
    () => new Model.CustomerClient(process.env.REACT_APP_API_URL),
    []
  )

  const tableRef = useRef<{ handleExport: () => void } | null>(null)
  const identity = useIdentity()

  const [editCustomer, setEditCustomer] = useState<Model.Customer>(new Model.Customer())
  const [isPropertyBarVisible, setIsPropertyBarVisible] = useState(false)
  const [filterTerms, setFilterTerms] = useState('')
  const [customers, setCustomers] = useState<Model.Customer[]>()
  const [isCustomerEditable, setIsCustomerEditable] = useState<boolean>(false)

  const updateCustomerProperty = (prop: string, value: string, type: string) => {
    switch (type) {
      case 'date':
        try {
          const date = moment.utc(value, 'MM/DD/YYYY').format()
          ;(editCustomer as any)[prop] = date
        } catch {}
        break

      default:
        ;(editCustomer as any)[prop] = value
        break
    }
  }

  const handleCancel = () => {
    setIsPropertyBarVisible(false)
    setTimeout(() => {
      setEditCustomer(new Model.Customer())
    }, 500)
  }

  const handleRowClick = (clickedCustomer: Model.Customer) => {
    setEditCustomer(Object.assign(new Model.Customer(), clickedCustomer))
    setIsPropertyBarVisible(true)
  }

  const handleUpdateCustomer = async (customer: Model.Customer) => {
    var dto = new Model.CustomerCreateOrUpdate()
    dto.name = customer.name
    dto.code = customer.code
    dto.firstName = customer.firstName
    dto.lastName = customer.lastName
    dto.address = customer.address
    dto.address2 = customer.address2
    dto.city = customer.city
    dto.state = customer.state
    dto.postalCode = customer.postalCode
    dto.emailAddress = customer.emailAddress
    dto.telephone = customer.telephone

    if (customer.uniqueID === undefined) {
      await customerApi.createCustomer(dto)
    } else {
      await customerApi.updateCustomer(customer.uniqueID, dto)
    }
    var customers = await customerApi.getCustomers()

    setCustomers(customers)
  }

  useEffect(() => {
    ;(async () => {
      try {
        const result = await customerApi.getCustomers()
        setCustomers(result)
      } catch (error) {
        Log.add(JSON.stringify(error))
      }
    })()
  }, [customerApi])

  useEffect(() => {
    if (identity) {
      setIsCustomerEditable(identity.isAdministrator())
    }
  }, [identity])

  return (
    customers && (
      <>
        <div className="header">Customers</div>
        <Menu>
          {isCustomerEditable ? (
            <MenuItem
              label="Add customer"
              icon="SquarePlus"
              onClick={() => {
                setEditCustomer(new Model.Customer())
                setIsPropertyBarVisible(true)
              }}
            />
          ) : (
            <></>
          )}
          {customers.length > 0 ? (
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
          id="customer-table"
          name="Customers"
          keyField="uniqueID"
          columns={columns}
          sourceData={customers}
          isPropertyBarVisible={isPropertyBarVisible}
          filterTerms={filterTerms}
          onRowClick={isCustomerEditable ? handleRowClick : null}></Table>
        {isCustomerEditable ? (
          <PropertyBar
            entityID={editCustomer.uniqueID ?? ''}
            isVisible={isPropertyBarVisible}
            onSave={() => handleUpdateCustomer(editCustomer)}
            onCancel={handleCancel}>
            <>
              <div className="caption">
                {editCustomer.uniqueID === undefined ? 'New customer' : 'Edit customer'}
              </div>
              {fields.map((o, i) => {
                return (
                  <TextInput
                    entityID={editCustomer.uniqueID?.toString()}
                    key={o.accessor}
                    type={o.type}
                    label={o.label}
                    name={o.accessor}
                    value={(editCustomer as any)[o.accessor]}
                    required={o.required ?? false}
                    regex={o.regex}
                    columnSpec={o.columnSpec}
                    formatting={o.formatting}
                    onChange={(value: string) =>
                      updateCustomerProperty(o.accessor, value, o.type)
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

export default Customers
