import { lowerCase, upperFirst } from 'lodash'

interface IPropertyTag {
  name?: string
}

const PropertyTag = ({ name }: IPropertyTag) => {
  return (
    name && <div className={`property-tag ${name}`}>{upperFirst(lowerCase(name))}</div>
  )
}

export default PropertyTag
